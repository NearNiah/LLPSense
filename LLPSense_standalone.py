import numpy as np
from pathlib import Path
import preprocess.misc as misc
from preprocess.utils import create_dir
from omegaconf import ListConfig
import hydra, pyrootutils
from omegaconf import DictConfig
import LLPSXG as llps
import itertools
from copy import deepcopy
from scipy.stats import linregress
from sparrow import Protein

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="LLPSense")
def main(cfg: DictConfig):
    feat_type = cfg.method.feat_type
    outputs = create_dir(Path(f'outputs/results/LLPSense_{cfg.expname}'))
    assert cfg.standalone.exp_task in ['test', 'predict', 'screen', 'mut', 'mut_inst', 'mut_screen']
    
    # model
    model = llps.LLPSXG(n_estimators=cfg.model.n_estimators, max_depth=cfg.model.max_depth, 
                        random_state=cfg.model.random_state, learning_rate=cfg.model.lr, min_child_weight=cfg.model.mcw,
                        colsample_bytree=cfg.model.col, subsample=cfg.model.sub, gamma=cfg.model.gamma, 
                        reg_lambda=cfg.model.reg_lambda, reg_alpha=cfg.model.reg_alpha, device=cfg.model.device)
    model.load_model(cfg.standalone.model_path)
    
    if cfg.standalone.exp_task == 'test':
        x_test, y_test, test_cond, test_group, test_name, test_seq = llps.extract(cfg.data.test_file, seq_length_min=cfg.data.seq_length_min, feat_type=feat_type, use_cond=True)
        merged_feature = np.concatenate((x_test, test_cond), axis=1)
        output = model.evaluate(merged_feature, y_test)
        print(f"Mean Accuracy: {output['accuracy']}, Mean AUROC: {output['roc_auc']}, Mean AUPRC: {output['pr_auc']}")
        llps.save_results(outputs / f"test.xlsx", output['prob'], roc_curve_data=output['roc_curve'], pr_curve_data=output['pr_curve'], cond=test_cond, seq=test_seq, name=test_name, y_true=y_test)
    
    elif cfg.standalone.exp_task == 'predict':
        x_test, y_test, test_cond, test_group, test_name, test_seq = llps.extract(cfg.data.test_file, seq_length_min=cfg.data.seq_length_min, feat_type=feat_type, use_cond=True)
        merged_feature = np.concatenate((x_test, test_cond), axis=1)
        output = model.predict_proba(merged_feature)[:, 1]
        llps.save_results(outputs / f"predict.xlsx", output, cond=test_cond, seq=test_seq, name=test_name)
    
    elif cfg.standalone.exp_task == 'screen':
        screen = cfg.standalone.screen
        apply_smooth = cfg.data.apply_smooth
        smoothing_window = cfg.data.smoothing_window
        
        # 2d screening, single condition monitoring
        if isinstance(screen, str):
            
            x_screen, y_screen, screen_cond, group_cond, name_screen, seq_screen = llps.extract_screen(cfg.data.screen_file, screen=screen, seq_length_min=cfg.data.seq_length_min, feat_type=feat_type)
            merged_feature = np.concatenate((x_screen, screen_cond), axis=1)
            output = model.predict_proba(merged_feature)[:, 1]
            
            # check irregular proteins
            if cfg.check.irregular:
                llps.check_irregular(output, screen)
            screen_cond_metric = llps.cond_to_metric(screen_cond)
            
            # draw screen graph
            indices_prot = sorted(np.unique(x_screen, return_index=True, axis=0)[1]) + [len(x_screen)]
            for i, index in enumerate(indices_prot):
                if index == len(x_screen): break
                start_index, end_index = index, indices_prot[i+1]
                seqname = name_screen[start_index]
                screen_plots = screen_cond_metric[start_index:end_index]
                prob_plots = output[start_index:end_index]
                
                if apply_smooth:
                    prob_plots = llps.moving_average(prob_plots, window_size=smoothing_window)
                llps.draw_screen_graph(screen_plots[:, misc.get_cond_index(screen)], prob_plots, screen, outputs / f"{screen}_{seqname}.png")
            
            # save raw data as excel
            llps.save_results(outputs / f"screen_{screen}.xlsx", output, cond=screen_cond, seq=seq_screen, name=name_screen)
            
        # 3d screening, multiple condition monitoring
        elif isinstance(screen, ListConfig):
            screen = list(screen)
            screen_pair = itertools.combinations(screen, 2)
            for screen_0, screen_1 in screen_pair:
                x_screen, y_screen, screen_cond, group_cond, name_screen, seq_screen = llps.extract_screen_mul(cfg.data.screen_file, screen=[screen_0, screen_1], seq_length_min=cfg.data.seq_length_min, feat_type=feat_type)
                merged_feature = np.concatenate((x_screen, screen_cond), axis=1)
                output = model.predict_proba(merged_feature)[:, 1]
                screen_cond_metric = llps.cond_to_metric(screen_cond)
                
                # draw screen graph
                indices_prot = sorted(np.unique(x_screen, return_index=True, axis=0)[1]) + [len(x_screen)]
                for i, index in enumerate(indices_prot):
                    if index == len(x_screen): break
                    start_index, end_index = index, indices_prot[i+1]
                    screen_plots = screen_cond_metric[start_index:end_index]
                    prob_plots = output[start_index:end_index]
                    screen_plots_0, screen_plots_1 = screen_plots[:, misc.get_cond_index(screen_0)], screen_plots[:, misc.get_cond_index(screen_1)]
                    graph_path = outputs / f"{screen_0}_{screen_1}_{i + 1}.png"
                    excel_path = outputs / f"{screen_0}_{screen_1}_{i + 1}_contour.xlsx"
                    llps.draw_screen_graph_mul(screen_plots_0, screen_plots_1, prob_plots, [screen_0, screen_1], graph_path, filepath_excel=excel_path, smooth=apply_smooth)
                
                # save raw data as excel
                llps.save_results(outputs / f"{screen_0}_{screen_1}.xlsx", output, cond=screen_cond, seq=seq_screen, name=name_screen)
        else:
            raise ValueError("Not supported screen condition.")
        
    elif cfg.standalone.exp_task == 'mut':
        # only support single protein
        
        # adjust this directly
        cond_control = [25/60, 0.1, 7.4/14] + [0, 0, 0, 0, 0, 0] + [0, 200/2000, 0] + [0] # temp, conc, pH, cagents(6), salts(3), glyc
        
        # start from here
        x_mut, y_mut, mut_cond, group_mut, name_mut, seq_mut, static_pos = llps.extract_mut(cfg.data.mut_file, cond=cond_control, seq_length_min=cfg.data.seq_length_min, feat_type=feat_type)
        merged_feature = np.concatenate((x_mut, mut_cond), axis=1)
        output = model.predict_proba(merged_feature)[:, 1]
        llps.save_results(outputs / f"mutation.xlsx", output, cond=mut_cond, seq=seq_mut, name=name_mut)
        wt_score, mut_score = output[0], output[1:]
        seq_len = len(seq_mut[0])
        original_seq = seq_mut[0]
        print("Wild Type Score: ", wt_score)
        
        valid_mask = np.zeros(seq_len)
        valid_mask[static_pos] = 1
        delta_high, delta_low = np.zeros(seq_len), np.zeros(seq_len)  # fill zero for non mutated position
        aa_high, aa_low = np.array(list(original_seq)), np.array(list(original_seq))
        for i in range(seq_len):
            if i in static_pos: continue
            
            # remove the same amino acid in misc.amino_acids
            amino_acids_filtered = deepcopy(misc.amino_acids)
            index_overlapped = amino_acids_filtered.index(original_seq[i])
            amino_acids_filtered.pop(index_overlapped)
            
            for j in range(19):
                
                # calculate delta score
                delta_score = mut_score[19*i+j] - wt_score
                if j == 0:
                    delta_high[i] = delta_score
                    delta_low[i] = delta_score
                    aa_high[i] = amino_acids_filtered[j]
                    aa_low[i] = amino_acids_filtered[j]
                else:
                    if delta_score > delta_high[i]:
                        delta_high[i] = delta_score
                        aa_high[i] = amino_acids_filtered[j]
                    elif delta_score < delta_low[i]:
                        delta_low[i] = delta_score
                        aa_low[i] = amino_acids_filtered[j]
        
        masked_delta_high = np.ma.masked_array(delta_high, mask=valid_mask)
        masked_delta_low = np.ma.masked_array(delta_low, mask=valid_mask)
        max_delta_pos, max_index_pos = masked_delta_high.max(), masked_delta_high.argmax()
        max_delta_neg, max_index_neg = masked_delta_low.min(), masked_delta_low.argmin()
        
        print("Most positive mutation:", max_delta_pos, seq_mut[0][max_index_pos]+str(max_index_pos+1)+aa_high[max_index_pos])
        print("Most negative mutation:", max_delta_neg, seq_mut[0][max_index_neg]+str(max_index_neg+1)+aa_low[max_index_neg])
        llps.draw_mut_graph(delta_high.tolist(), delta_low.tolist(), outputs / f"mut.png")
        
    elif cfg.standalone.exp_task == 'mut_screen':
        
        # only support single screening, single protein
        screen = cfg.standalone.screen
        apply_smooth = cfg.data.apply_smooth
        smoothing_window = cfg.data.smoothing_window
        assert isinstance(screen, str)  # only support single screen condition
        assert cfg.standalone.mut_direction in ['positive', 'negative']
        
        x_mut, y_mut, mut_cond, group_mut, name_mut, seq_mut, static_pos = llps.extract_mut(cfg.data.mut_file, cond=screen, seq_length_min=cfg.data.seq_length_min, feat_type=feat_type)
        merged_feature = np.concatenate((x_mut, mut_cond), axis=1)
        output = model.predict_proba(merged_feature)[:, 1]
        screen_cond_metric = llps.cond_to_metric(mut_cond)
        seq_len = len(seq_mut[0])
        original_seq = seq_mut[0]
        
        # draw screen graph
        mut_scores = []
        indices_prot = sorted(np.unique(x_mut, return_index=True, axis=0)[1]) + [len(x_mut)]
        for i, index in enumerate(indices_prot):
            if index == len(x_mut): break
            start_index, end_index = index, indices_prot[i+1]
            screen_plots = screen_cond_metric[start_index:end_index]
            prob_plots = output[start_index:end_index]
            
            if apply_smooth:
                prob_plots = llps.moving_average(prob_plots, window_size=smoothing_window)
            
            if index == 0: 
                wt_scores = prob_plots
            else:
                mut_scores.append(prob_plots)
        mut_scores = np.stack(mut_scores, axis=0)
                
        # determine best mutation position
        valid_mask = np.zeros(seq_len, dtype=bool)
        valid_mask[static_pos] = True
        xvalues = screen_cond_metric[:len(wt_scores), misc.get_cond_index(screen)]
        delta_slopes = np.full(seq_len, np.nan)
        delta_aa = np.array(list(original_seq))
        for i in range(seq_len):
            if i in static_pos: continue
            
            # remove the same amino acid in misc.amino_acids
            amino_acids_filtered = deepcopy(misc.amino_acids)
            index_overlapped = amino_acids_filtered.index(original_seq[i])
            amino_acids_filtered.pop(index_overlapped)
            
            for j in range(19):
                
                # calculate delta score
                delta = mut_scores[19*i+j] - wt_scores  # shape: (num_sreen)
                delta_slope = linregress(xvalues, delta)[0]
                
                if j == 0:
                    delta_slopes[i] = delta_slope
                    delta_aa[i] = amino_acids_filtered[j]
                else:
                    if cfg.standalone.mut_direction == 'negative':
                        if delta_slope < delta_slopes[i]:
                            delta_slopes[i] = delta_slope
                            delta_aa[i] = amino_acids_filtered[j]
                    else:
                        if delta_slope > delta_slopes[i]:
                            delta_slopes[i] = delta_slope
                            delta_aa[i] = amino_acids_filtered[j]
        
        # get best positions (top-k)
        masked_delta_sorted = np.ma.masked_array(delta_slopes, mask=valid_mask)
        if cfg.standalone.mut_direction == 'negative':
            selected_indices = masked_delta_sorted.argsort(endwith=True)[:cfg.standalone.num_mut]
        else:
            selected_indices = masked_delta_sorted.argsort(endwith=False)[::-1][:cfg.standalone.num_mut]
        selected_values = masked_delta_sorted[selected_indices].data
        selected_aas = delta_aa[selected_indices]
        
        # save with mutated protein
        mutated_seq = np.array(list(original_seq))
        mutated_seq[selected_indices] = selected_aas
        mutated_seq = ''.join(mutated_seq)
        print_str = f"{seq_mut[0]} -> {mutated_seq} : "
        for i in range(len(selected_indices)):
            mutation_mark = f"{original_seq[selected_indices[i]]}{selected_indices[i]+1}{selected_aas[i]} with {selected_values[i]:.4f}"
            print_str += f"{mutation_mark}  "
            static_pos.append(selected_indices[i])
        print("==================== Mutation Results ===================")
        print(print_str)
        
        llps.save_mutated_protein(cfg.data.mut_file, mutated_seq, static_pos)
    else:
        raise ValueError("Not supported experiment task type.")
        
    print("result files saved successfully.")
    print("Done! Thank you for using LLPSense.")

if __name__ == '__main__':
    main()



