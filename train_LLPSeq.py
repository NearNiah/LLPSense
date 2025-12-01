from omegaconf import DictConfig
import pyrootutils
import numpy as np
import hydra, optuna, wandb
import LLPSXG as llps
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from pathlib import Path
from preprocess.utils import create_dir


minlogscore = float('inf')
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 

def calc_weights(train_no, id_no, noid_no, non_no):
    
    if model_weight:  
        weight_id = train_no / id_no
        weight_noid = train_no / noid_no
        weight_non = train_no / non_no
    else:
        weight_id = 1
        weight_noid = 1
        weight_non = 1
    
    weights = []

    for i in range(id_no):
        weights.append(weight_id)
    for i in range(noid_no):
        weights.append(weight_noid)
    for i in range(non_no):
        weights.append(weight_non)
        
    weight = np.array(weights)
    
    return weight


def objective(trial, x_train, y_train, weight, id_no, noid_no, non_no):
    global minlogscore
    
    # Define the search space for hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.00001, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.00001, 100.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.00001, 100.0, log=True)
    }
    
    ptype = []
    for i in range(id_no):
        ptype.append(0)
    for i in range(noid_no):
        ptype.append(1)
    for i in range(non_no):
        ptype.append(2)
    ptype = np.array(ptype)
    
    # Perform 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auroc_scores = []
    logloss_scores = []
    for train_index, val_index in skf.split(x_train, ptype):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        weight_fold = weight[train_index]
        
        model = llps.LLPSXG(**params, device=device)
        
        # Fit the model
        model.fit(x_train_fold, y_train_fold, weight_fold)
        
        # Predict probabilities on validation fold for each model and average the scores
        y_val_pred_proba = model.cal_prob(x_val_fold)

        # Calculate auroc, logloss score for this fold
        auroc_score = roc_auc_score(y_val_fold, y_val_pred_proba)
        auroc_scores.append(auroc_score)
        logloss_score = log_loss(y_val_fold, y_val_pred_proba)
        logloss_scores.append(logloss_score)

    model = llps.LLPSXG(**params, device=device)
    model.fit(x_train, y_train, weight)
        
    aurocscore = np.mean(auroc_scores) 
    logscore = np.mean(logloss_scores)
    
    if logscore < minlogscore:
        minlogscore = logscore
    if wandb_mode:
        params['AUROC'] = aurocscore
        params['LOGLOSS'] = logscore
        params['MINLOSS'] = minlogscore
        wandb.log(params)

    # Return the mean logloss score across all folds
    return logscore



def validation(cfg, outputs_dir, x_train, y_train, weight, id_no, noid_no, non_no):
    
    params = {
        'n_estimators': cfg.model.n_estimators,
        'max_depth': cfg.model.max_depth,
        'learning_rate': cfg.model.lr,
        'min_child_weight': cfg.model.mcw,
        'colsample_bytree': cfg.model.col,
        'subsample': cfg.model.sub,
        'gamma': cfg.model.gamma,
        'reg_lambda': cfg.model.reg_lambda,
        'reg_alpha': cfg.model.reg_alpha
    }
    
    ptype = []
    for i in range(id_no):
        ptype.append(0)
    for i in range(noid_no):
        ptype.append(1)
    for i in range(non_no):
        ptype.append(2)
    ptype = np.array(ptype)
    
    # Perform 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auroc_scores = []
    logloss_scores = []
    for fold_idx, (train_index, val_index) in enumerate(skf.split(x_train, ptype)):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        weight_fold = weight[train_index]
        
        model = llps.LLPSXG(**params, device=device)
        
        # Fit the model
        model.fit(x_train_fold, y_train_fold, weight_fold)
        
        # Predict probabilities on validation fold for each model and average the scores
        y_val_pred_proba = model.cal_prob(x_val_fold)

        # Calculate auroc, logloss score for this fold
        auroc_score = roc_auc_score(y_val_fold, y_val_pred_proba)
        auroc_scores.append(auroc_score)
        logloss_score = log_loss(y_val_fold, y_val_pred_proba)
        logloss_scores.append(logloss_score)
        
        # save model and clusters
        clusters = np.unique(np.array(ptype)[val_index])
        model.save_model(str(outputs_dir / f"LLPSeq_{cfg.method.feat_type}_fold_{fold_idx+1}.pkl"))
        np.save(outputs_dir / f"LLPSeq_{cfg.method.feat_type}_fold_{fold_idx+1}_cluster.npy", clusters)
    
    aurocscore = np.mean(auroc_scores) 
    logscore = np.mean(logloss_scores)
    
    # print
    print(f"Mean Loss: {logscore}, Mean AUROC: {aurocscore}")
    
    return logscore, aurocscore



@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="LLPSeq")
def main(cfg: DictConfig):
    global device, model_weight, wandb_mode
    device = cfg.model.device
    model_weight = cfg.model.weight
    wandb_mode = cfg.method.wandb
    results_dir = create_dir(Path('outputs/results'))
    outputs_dir = create_dir(Path('outputs/models'))
    log_dir = create_dir(Path('outputs/logs'))

    x_train, y_train = llps.extract(cfg.data.train_file, seq_length_min=cfg.data.seq_length_min, feat_type=cfg.method.feat_type, use_cond=False)
    
    train_no = len(y_train)
    id_no = 195
    noid_no = 64
    non_no = train_no - id_no - noid_no
    
    weight = calc_weights(train_no, id_no, noid_no, non_no)

    if cfg.method.mode == 'optuna':
        if wandb_mode:
            wandb.init(entity='llpsense', project='LLPSense') # put your wandb entity and project name here

        study = optuna.create_study(direction='minimize')

        # Start the optimization
        study.optimize(lambda trial: objective(trial, x_train, y_train, weight, id_no, noid_no, non_no), n_trials=1000)
        
        # Get the best hyperparameters found by Optuna
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
        
        model = llps.LLPSXG(**best_params, device=device)
        model.fit(x_train, y_train, weight)
    elif cfg.method.mode == 'valid':
        validation(cfg, log_dir, x_train, y_train, weight, id_no, noid_no, non_no)
    else:
        model = llps.LLPSXG(n_estimators=cfg.model.n_estimators, max_depth=cfg.model.max_depth, 
                    random_state=cfg.model.random_state, learning_rate=cfg.model.lr, min_child_weight=cfg.model.mcw,
                    colsample_bytree=cfg.model.col, subsample=cfg.model.sub, gamma=cfg.model.gamma, 
                    reg_lambda=cfg.model.reg_lambda, reg_alpha=cfg.model.reg_alpha, device=cfg.model.device)
        model.fit(x_train, y_train, weight)

    print("Training complete.")
    
    if cfg.check.test_score and cfg.method.mode != 'valid':

        x_test_id, y_test_id = llps.extract(cfg.data.test_file1, seq_length_min=cfg.data.seq_length_min, feat_type=cfg.method.feat_type, use_cond=False)
        x_test_noid, y_test_noid = llps.extract(cfg.data.test_file2, seq_length_min=cfg.data.seq_length_min, feat_type=cfg.method.feat_type, use_cond=False)
        
        # Evaluate the model
        metrics_id = model.evaluate(x_test_id, y_test_id)
        metrics_noid = model.evaluate(x_test_noid, y_test_noid)

        # Print the evaluation metrics
        print("Accuracy for ID:", metrics_id['accuracy'], "/ Accuracy for noID:", metrics_noid['accuracy'])
        print("AUROC for ID:", metrics_id['roc_auc'], "/ AUROC for noID:", metrics_noid['roc_auc'])
        print("AUPRC for ID:", metrics_id['pr_auc'], "/ AUPRC for noID:", metrics_noid['pr_auc'])
        
        calc_results_id = model.cal_prob(x_test_id) 
        calc_results_noid = model.cal_prob(x_test_noid)

        # Save the evaluation results to an Excel file
        llps.save_results(str(results_dir / f"LLPSeq_{cfg.expname}_ID.xlsx"), calc_results_id, 
                        metrics_id['roc_curve'], metrics_id['pr_curve'], y_true=y_test_id)
        llps.save_results(str(results_dir / f"LLPSeq_{cfg.expname}_noID.xlsx"), calc_results_noid, 
                        metrics_noid['roc_curve'], metrics_noid['pr_curve'], y_true=y_test_noid)
    
    if cfg.check.save_model and cfg.method.mode != 'valid':
        model.save_model(str(outputs_dir / f"LLPSeq_{cfg.expname}.pkl"))



if __name__ == '__main__':
    main()