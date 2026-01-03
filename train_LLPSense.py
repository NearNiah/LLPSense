import xgboost as xgb
from omegaconf import DictConfig
import numpy as np
import hydra, pyrootutils, optuna, wandb
import LLPSXG as llps
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, average_precision_score, roc_curve, auc, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
from pathlib import Path
from preprocess.utils import create_dir
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


minlogscore = float('inf')
maxaucscore = 0
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True) 
 
    
def objective(trial, merged_feature, y_train, train_cluster):
    global minlogscore, maxaucscore 
    
    # Define the search space for hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.00001, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.00001, 100.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.00001, 100.0, log=True)
    }

    # Perform 5-fold cross-validation
    gkf = GroupKFold(n_splits=5)
    logloss_scores, accuracy_scores, auroc_scores, auprc_scores = [], [], [], []
    for train_index, val_index in gkf.split(merged_feature, y_train, groups=train_cluster):
        x_train_fold, x_val_fold = merged_feature[train_index], merged_feature[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = llps.LLPSXG(**params, device=device)
        
        # Fit the model
        model.fit(x_train_fold, y_train_fold)
        
        # Predict probabilities on validation fold for each model and average the scores
        y_val_pred_proba = model.cal_prob(x_val_fold)
        y_pred = model.model.predict(x_val_fold)

        # Calculate AUROC score for this fold
        log_loss_score = log_loss(y_val_fold, y_val_pred_proba)
        logloss_scores.append(log_loss_score)
        accuracy = accuracy_score(y_val_fold, y_pred)
        accuracy_scores.append(accuracy)
        auroc_score = roc_auc_score(y_val_fold, y_val_pred_proba)
        auroc_scores.append(auroc_score)
        auprc_score = average_precision_score(y_val_fold, y_val_pred_proba)
        auprc_scores.append(auprc_score)
    
    logscore = np.mean(logloss_scores)
    accuracyscore = np.mean(accuracy_scores)
    aucscore = np.mean(auroc_scores) 
    prcscore = np.mean(auprc_scores)    
    
    if maxaucscore < aucscore:
        maxaucscore = aucscore
        
    if wandb_mode:
        params['AUROC'] = aucscore
        params['ACCURACY'] = accuracyscore
        params['LOGLOSS'] = logscore
        params['MINLOSS'] = minlogscore
        params['AUPRC'] = prcscore
        
        wandb.log(params)
    
    return aucscore


def validation(cfg, outputs_dir, merged_feature, y_train, train_cluster):
    
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
    
    # Perform 5-fold cross-validation
    gkf = GroupKFold(n_splits=5)
    mean_fpr = np.linspace(0, 1, 1000)
    mean_recall = np.linspace(0, 1, 1000)
    disp_roc, disp_pr, tprs, precision_list = [], [], [], []
    logloss_scores, accuracy_scores, auroc_scores, auprc_scores = [], [], [], []
    for fold_idx, (train_index, val_index) in enumerate(gkf.split(merged_feature, y_train, groups=train_cluster)):
        x_train_fold, x_val_fold = merged_feature[train_index], merged_feature[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = llps.LLPSXG(**params, device=cfg.model.device)
        
        # Fit the model
        model.fit(x_train_fold, y_train_fold)
        
        # Predict probabilities on validation fold for each model and average the scores
        y_val_pred_proba = model.cal_prob(x_val_fold)
        y_pred = model.model.predict(x_val_fold)

        # Calculate AUROC score for this fold
        log_loss_score = log_loss(y_val_fold, y_val_pred_proba)
        logloss_scores.append(log_loss_score)
        accuracy = accuracy_score(y_val_fold, y_pred)
        accuracy_scores.append(accuracy)
        auroc_score = roc_auc_score(y_val_fold, y_val_pred_proba)
        auroc_scores.append(auroc_score)
        auprc_score = average_precision_score(y_val_fold, y_val_pred_proba)
        auprc_scores.append(auprc_score)
        
        # Get ROC and PR curves
        fpr, tpr, _ = roc_curve(y_val_fold, y_val_pred_proba)
        disp_roc.append(RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr), estimator_name=f"Fold {fold_idx}"))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        precision, recall, _ = precision_recall_curve(y_val_fold, y_val_pred_proba)
        disp_pr.append(PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name=f"PR fold {fold_idx}"))
        precision_interp = interp1d(recall, precision, kind='linear', fill_value="extrapolate")
        interp_precision = precision_interp(mean_recall)
        interp_precision[0] = 1.0
        precision_list.append(interp_precision)
        
        # save model and clusters
        clusters = np.unique(np.array(train_cluster)[val_index])
        model.save_model(str(outputs_dir / f"LLPSense_{cfg.method.feat_type}_fold_{fold_idx+1}.pkl"))
        np.save(outputs_dir / f"LLPSense_{cfg.method.feat_type}_fold_{fold_idx+1}_cluster.npy", clusters)
    
    logscore = np.mean(logloss_scores)
    accuracyscore = np.mean(accuracy_scores)
    aucscore = np.mean(auroc_scores) 
    prcscore = np.mean(auprc_scores)
    std_accuracy = np.std(accuracy_scores)
    
    # print
    print(f"Mean Loss: {logscore}, Mean Accuracy: {accuracyscore}, Mean AUROC: {aucscore}, Mean AUPRC: {prcscore}, Std Accuracy: {std_accuracy}")

    # plot ROC and PR curves
    llps.draw_cross_roc_curve(mean_fpr, tprs, auroc_scores, disp_roc, str(outputs_dir / 'llpsense_roc_curve_fold.png'))
    llps.draw_cross_pr_curve(mean_recall, precision_list, auprc_scores, disp_pr, str(outputs_dir / 'llpsense_pr_curve_fold.png'))
    
    return logscore, accuracyscore, aucscore, prcscore


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="LLPSense")
def main(cfg: DictConfig):
    global device, wandb_mode
    assert cfg.method.mode in ['optuna', 'valid', 'train']
    device = cfg.model.device
    wandb_mode = cfg.method.wandb
    results_dir = create_dir(Path('outputs/results'))
    outputs_dir = create_dir(Path('outputs/models'))
    log_dir = create_dir(Path('outputs/logs'))
    
    x_train, y_train, train_cond, train_cluster, train_seq, train_name = llps.extract(cfg.data.train_file, seq_length_min=cfg.data.seq_length_min, feat_type=cfg.method.feat_type, use_cond=True)
        
    merged_feature = np.concatenate((x_train, train_cond), axis=1)

    if cfg.method.mode == 'optuna':
        if wandb_mode:
            wandb.init(entity='llpsense', project='LLPSense')
        
        study = optuna.create_study(direction='maximize')
        
        # Start the optimization
        study.optimize(lambda trial: objective(trial, merged_feature, y_train, train_cluster), n_trials=1000)
        
        # Get the best hyperparameters found by Optuna
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
        
        # Initialize the model with the best hyperparameters
        model = llps.LLPSXG(**best_params, device=cfg.model.device)
        model.fit(merged_feature, y_train)
    elif cfg.method.mode == 'valid':
        validation(cfg, log_dir, merged_feature, y_train, train_cluster)
    else:
        model = llps.LLPSXG(n_estimators=cfg.model.n_estimators, max_depth=cfg.model.max_depth, 
                            random_state=cfg.model.random_state, learning_rate=cfg.model.lr, min_child_weight=cfg.model.mcw,
                            colsample_bytree=cfg.model.col, subsample=cfg.model.sub, gamma=cfg.model.gamma, 
                            reg_lambda=cfg.model.reg_lambda, reg_alpha=cfg.model.reg_alpha, device=cfg.model.device)
        model.fit(merged_feature, y_train)
    
    print("Training complete.")
    
    if cfg.check.test_score and cfg.method.mode != 'valid':    
        x_test, y_test, test_cond, test_cluster, test_seq, test_name = llps.extract(cfg.data.test_file, seq_length_min=cfg.data.seq_length_min, feat_type=cfg.method.feat_type, use_cond=True)
        merged_feature_test = np.concatenate((x_test, test_cond), axis=1)
        
        evaluation_results = model.evaluate(merged_feature_test, y_test)
            
        print("AUROC for test set:", evaluation_results['roc_auc'], "/ AUPRC for test set:", evaluation_results['pr_auc'], "/ Accuracy for test set:", evaluation_results['accuracy'])
        
        # Save the evaluation results to an Excel file
        calc_results = model.cal_prob(merged_feature_test, y_test) 
        llps.save_results(str(results_dir / f"Pred_Prob_{cfg.expname}.xlsx"), calc_results, 
                         evaluation_results['roc_curve'], evaluation_results['pr_curve'], test_cond, test_seq, y_true=y_test)
    
    if cfg.check.save_model and cfg.method.mode != 'valid':
        model.save_model(str(outputs_dir / f"LLPSense_{cfg.expname}.pkl"))
    
    if cfg.check.feature_importance and cfg.method.mode != 'valid':
        booster_load = model.model.get_booster()
        xgb.plot_importance(booster_load, max_num_features=20)
        plt.show()


if __name__ == '__main__':
    main()