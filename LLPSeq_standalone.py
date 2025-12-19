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

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 

@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="LLPSeq")
def main(cfg: DictConfig):
    feat_type = cfg.method.feat_type
    outputs = create_dir(Path(f'outputs/results/LLPSeq_{cfg.expname}'))
    
    # model
    model = llps.LLPSXG(n_estimators=cfg.model.n_estimators, max_depth=cfg.model.max_depth, 
                        random_state=cfg.model.random_state, learning_rate=cfg.model.lr, min_child_weight=cfg.model.mcw,
                        colsample_bytree=cfg.model.col, subsample=cfg.model.sub, gamma=cfg.model.gamma, 
                        reg_lambda=cfg.model.reg_lambda, reg_alpha=cfg.model.reg_alpha, device=cfg.model.device)
    model.load_model(cfg.standalone.model_path)
    
    x_test, y_test, _, test_name, test_seq = llps.extract(cfg.data.test_file1, seq_length_min=cfg.data.seq_length_min, feat_type=cfg.method.feat_type, use_cond=False)
    
    y_pred = model.cal_prob(x_test)
    llps.save_results(outputs / f"predict.xlsx", y_pred, seq=test_seq, name=test_name)
    
    
if __name__ == '__main__':
    main()
