
import hydra, pyrootutils
from omegaconf import DictConfig
import numpy as np
from pathlib import Path
import LLPSXG as llps
from copy import deepcopy
import preprocess.misc as misc
from einops import repeat
from preprocess.utils import create_dir
from collections import OrderedDict

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="LLPSense")
def main(cfg: DictConfig):
    
    nsplits = cfg.thresholds.nsplits
    feat_type = cfg.method.feat_type
    device = cfg.model.device
    apply_smooth = cfg.data.apply_smooth
    smoothing_window = cfg.data.smoothing_window
    screen = cfg.thresholds.screen
    srcfile = cfg.thresholds.test_file
    modelfile = Path('outputs/logs')
    outputs = create_dir(Path(f'outputs/trends_{feat_type}'))
    
    # load clusters
    clusters = [np.load(str(modelfile / f"LLPSense_{feat_type}_fold_{i+1}_cluster.npy"), allow_pickle=True).tolist() for i in range(nsplits)]
    x_test, y_test, test_cond, test_cluster, test_seq, test_name = llps.extract(srcfile, seq_length_min=cfg.data.seq_length_min, feat_type=cfg.method.feat_type, use_cond=True)
    
    # model
    model = llps.LLPSXG(n_estimators=cfg.model.n_estimators, max_depth=cfg.model.max_depth, 
                        random_state=cfg.model.random_state, learning_rate=cfg.model.lr, min_child_weight=cfg.model.mcw,
                        colsample_bytree=cfg.model.col, subsample=cfg.model.sub, gamma=cfg.model.gamma, 
                        reg_lambda=cfg.model.reg_lambda, reg_alpha=cfg.model.reg_alpha, device=device)
    
    fold = []
    for value in test_cluster:
        for n, cluster_value in enumerate(clusters):
            if value in cluster_value:
                fold.append(n)
                break
    fold_unique = np.unique(fold).tolist()
    
    x_interp_points, x_diff_points = {}, {}
    for fold_idx in fold_unique:
        model.load_model(str(modelfile / f"LLPSense_{feat_type}_fold_{fold_idx+1}.pkl"))
        indices = list(np.where(np.array(fold) == fold_idx)[0])
        
        for index in indices:
            
            x_test_fold = x_test[index]
            y_test_fold = y_test[index]
            test_cond_fold = test_cond[index]
            test_cond_folds, test_cond_folds_plot = [], []
            if screen == 'temp':
                thres_gt = test_cond_fold[0] * misc.max_temp
                for i in range(61):
                    test_cond_folds.append(deepcopy(test_cond_fold))
                    test_cond_folds[-1][0] = i / misc.max_temp
                    test_cond_folds_plot.append(test_cond_folds[-1][0] * misc.max_temp)
            elif screen == 'nacl':
                thres_gt = test_cond_fold[10] * misc.max_temp
                for i in range(1, 101):
                    test_cond_folds.append(deepcopy(test_cond_fold))
                    test_cond_folds[-1][10] = 10 * i / misc.max_nacl
                    test_cond_folds_plot.append(test_cond_folds[-1][10] * misc.max_nacl)
            elif screen == 'pH':
                thres_gt = test_cond_fold[2] * misc.max_temp
                for i in range(61):
                    test_cond_folds.append(deepcopy(test_cond_fold))
                    test_cond_folds[-1][2] = (4 + i * 0.1) / misc.max_pH
                    test_cond_folds_plot.append(test_cond_folds[-1][2] * misc.max_pH)
            elif screen == 'conc':
                thres_gt = test_cond_fold[1] * misc.max_temp
                for i in range(1, 201):
                    conc_value = test_cond_fold[1] / 100 * i
                    if conc_value > 1: break
                    test_cond_folds.append(deepcopy(test_cond_fold))
                    test_cond_folds[-1][1] = conc_value
                    test_cond_folds_plot.append(test_cond_folds[-1][1] * misc.max_conc)
                    
            # estimate the trend
            x_test_fold = repeat(x_test_fold, 'c -> n c', n=len(test_cond_folds))
            test_cond_folds = np.stack(test_cond_folds, axis=0)
            merged_feature_test = np.concatenate((x_test_fold, test_cond_folds), axis=1)
            prob = model.predict_proba(merged_feature_test)[:, 1]
            if apply_smooth:
                prob = llps.moving_average(prob, window_size=smoothing_window)
            xvalue = np.array(test_cond_folds_plot)
            
            # draw graph and save
            llps.draw_screen_graph(xvalue, prob, screen, outputs / f"{index + 1}_{screen}_{y_test_fold}.png")
            
            # get intersection points
            score = np.sign(prob - 0.5)
            score_diff_pts = np.argwhere(np.abs(np.diff(score, axis=0)) >= 1)
            prob_start, prob_end = prob[score_diff_pts], prob[score_diff_pts + 1]
            xvalue_start, xvalue_end = xvalue[score_diff_pts], xvalue[score_diff_pts + 1]
            x_interp = xvalue_start + (0.5 - prob_start) * (xvalue_end - xvalue_start) / (prob_end - prob_start)
            
            # save the intersection points
            x_interp_points[index + 1] = x_interp
            x_diff_points[index + 1] = np.abs(x_interp - thres_gt)
            
    # print the intersection points
    x_interp_points = OrderedDict(sorted(x_interp_points.items()))
    for key, value in x_interp_points.items():
        print(f"Index: {key}, Intersection points: {value}, Difference: {x_diff_points[key]}")
    
    

if __name__ == "__main__":
    main()
