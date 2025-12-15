import re
import csv
import importlib
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
import pandas as pd
import xgboost as xgb
import preprocess.misc as misc
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
import h5py, joblib
import scipy.signal
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


# Set the font size for the plots
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 32


class LLPSXG:
    def __init__(self, n_estimators, max_depth, learning_rate, min_child_weight, colsample_bytree, subsample, gamma, reg_lambda, reg_alpha,
                 objective='binary:logistic', random_state=42, device='cuda:0', eval_metric='logloss', booster='gbtree', sampling_method='uniform'):
        self.model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, objective=objective, 
                                        learning_rate=learning_rate, random_state=random_state, gamma=gamma,
                                        min_child_weight=min_child_weight, colsample_bytree=colsample_bytree, subsample=subsample,
                                        reg_lambda=reg_lambda, reg_alpha=reg_alpha, 
                                        device=device, eval_metric=eval_metric, booster=booster, sampling_method=sampling_method) 
        
        self.encoder = None        

    def encoding(self, X, y, fit_encoder=False):
        X_transformed = self.model.apply(X)
        if fit_encoder:
            self.encoder = OneHotEncoder().fit(X_transformed)
        
    def fit(self, X_train, y_train, weight=None):
        self.model.fit(X_train, y_train, sample_weight=weight)
        self.encoding(X_train, y_train, fit_encoder=True)
        
    def predict_proba(self, x):
        return self.model.predict_proba(x)
        
    def evaluate(self, X_val, y_val):
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        return {'accuracy': accuracy, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 
                'roc_curve': {'fpr': fpr, 'tpr': tpr},'pr_curve': {'precision': precision, 'recall': recall}, 'prob': y_pred_proba}
    
    def cal_prob(self, X_val, y_val=None):
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]  # Prediction probabilities for positive class
        return y_pred_proba
    
    def save_model(self, filepath):
        # Store both the model and the encoder in a dictionary
        model_dict = {'model': self.model, 'encoder': self.encoder}
        # Use joblib to save the dictionary to the specified filepath
        joblib.dump(model_dict, filepath)
        
    def load_model(self, filepath):
        # Load the dictionary from the specified filepath
        model_dict = joblib.load(filepath)
        # Retrieve the model and encoder from the dictionary
        self.model = model_dict['model']
        self.encoder = model_dict['encoder']


        
def save_results(filepath, y_pred_proba, roc_curve_data=None, pr_curve_data=None, cond=None, seq=None, name=None, y_true=None):
    # Create a DataFrame to store prediction probabilities
    
    if cond is not None:
        temp = [sublist[0]*misc.max_temp for sublist in cond]
        conc = [sublist[1]*misc.max_conc for sublist in cond]
        pH = [sublist[2]*misc.max_pH for sublist in cond]
        peg1 = [sublist[3]*misc.max_cagent for sublist in cond]
        peg2 = [sublist[4]*misc.max_cagent for sublist in cond]
        peg3 = [sublist[5]*misc.max_cagent for sublist in cond]
        ficoll = [sublist[6]*misc.max_cagent for sublist in cond]
        dextran1 = [sublist[7]*misc.max_cagent for sublist in cond]
        dextran2 = [sublist[8]*misc.max_cagent for sublist in cond]
        mgcl2 = [sublist[9]*misc.max_mgcl2 for sublist in cond]
        nacl = [sublist[10]*misc.max_nacl for sublist in cond]
        kcl = [sublist[11]*misc.max_kcl for sublist in cond]
        glyc = [sublist[12]*misc.max_glyc for sublist in cond]
    
        df_results = pd.DataFrame({'Protein_Name': name,
                                    'Sequence': seq,
                                    'Temperature': temp,
                                    'Concentration': conc,
                                    'pH': pH,
                                    'PEG 300~1000': peg1,
                                    'PEG 3000~6000': peg2,
                                    'PEG 8000~20000': peg3,
                                    'Ficoll': ficoll,
                                    'Dextran ~40': dextran1,
                                    'Dextran 70~': dextran2,
                                    'MgCl2': mgcl2,
                                    'NaCl': nacl,
                                    'KCl': kcl,
                                    'glycerol': glyc,
                                    'Predicted_Probability': y_pred_proba,
                                    })
    else:
        df_results = pd.DataFrame({'Predicted_Probability': y_pred_proba})
    
    if y_true is not None:
        df_results['True_Label'] = y_true
        
    if roc_curve_data is None and pr_curve_data is None:
        # Save the DataFrame to an Excel file
        df_results.to_excel(filepath, index=False, sheet_name='Results')
        
    else:
        with pd.ExcelWriter(filepath) as writer:
            df_results.to_excel(writer, index=False, sheet_name='Results')
            
            if roc_curve_data is not None:
                # Write ROC curve data
                roc_curve_df = pd.DataFrame({'FPR': roc_curve_data['fpr'], 'TPR': roc_curve_data['tpr']})
                roc_curve_df.to_excel(writer, index=False, sheet_name='ROC_Curve')
                
            if pr_curve_data is not None:
                # Write PR curve data
                pr_curve_df = pd.DataFrame({'Precision': pr_curve_data['precision'], 'Recall': pr_curve_data['recall']})
                pr_curve_df.to_excel(writer, index=False, sheet_name='PR_Curve')
                
                
def save_mut_results(filepath, score_ori, score_delta, seqs_ori, seqs_mut, names_ori, names_mut, srcfilename, cond=None, y_true=None):
    
    # Create a DataFrame to store prediction probabilities
    df_results = pd.DataFrame({'ProtName_WT': names_ori,
                               'ProtName_Mut': names_mut,
                               'Sequence_WT': seqs_ori,
                               'Sequence_Mut': seqs_mut,
                               'Probability_WT': score_ori,
                               'Probability_Delta': score_delta,
                               'Source_File': srcfilename
                                })
    
    if y_true is not None:
        df_results['True_Label'] = y_true
    
    if cond is not None:
        temp = [sublist[0]*misc.max_temp for sublist in cond]
        conc = [sublist[1]*misc.max_conc for sublist in cond]
        pH = [sublist[2]*misc.max_pH for sublist in cond]
        peg1 = [sublist[3]*misc.max_cagent for sublist in cond]
        peg2 = [sublist[4]*misc.max_cagent for sublist in cond]
        peg3 = [sublist[5]*misc.max_cagent for sublist in cond]
        ficoll = [sublist[6]*misc.max_cagent for sublist in cond]
        dextran1 = [sublist[7]*misc.max_cagent for sublist in cond]
        dextran2 = [sublist[8]*misc.max_cagent for sublist in cond]
        mgcl2 = [sublist[9]*misc.max_mgcl2 for sublist in cond]
        nacl = [sublist[10]*misc.max_nacl for sublist in cond]
        kcl = [sublist[11]*misc.max_kcl for sublist in cond]
        glyc = [sublist[12]*misc.max_glyc for sublist in cond]
        
        df_results['Temperature'] = temp
        df_results['Concentration'] = conc
        df_results['pH'] = pH
        df_results['PEG 300~1000'] = peg1
        df_results['PEG 3000~6000'] = peg2
        df_results['PEG 8000~20000'] = peg3
        df_results['Ficoll'] = ficoll
        df_results['Dextran ~40'] = dextran1
        df_results['Dextran 70~'] = dextran2
        df_results['MgCl2'] = mgcl2
        df_results['NaCl'] = nacl
        df_results['KCl'] = kcl
        df_results['glycerol'] = glyc
        
    # Save the DataFrame to an Excel file
    df_results.to_excel(filepath, index=False, sheet_name='Results')


def extract(path, seq_length_min=None, feat_type='t5', use_cond=False):
    protfeat, seq, cond_set, label, group, name = [], [], [], [], [], []
    data = np.load(path, allow_pickle=True)['data']
    for sample in tqdm(data, desc='Loading features from data'):
        sample_dict = edict(sample)
        if seq_length_min is not None:
            if len(sample_dict.seq) < seq_length_min: continue
        with h5py.File(sample_dict[f'{feat_type}_feature_path'], 'r') as f:
            fvalue = np.array(f['protein_feat'][0][:])
            flist = fvalue.tolist()
            protfeat.append(flist)
        label.append(sample_dict.score)
        seq.append(sample_dict.seq)
        group.append(sample_dict.group)
        name.append(sample_dict.name)
        
        if use_cond:
            cond = [sample_dict.temp, sample_dict.conc, sample_dict.pH] + sample_dict.cagent.tolist() + sample_dict.salt.tolist() + [sample_dict.glyc]
            cond_set.append(cond)
    
    if use_cond:
        return np.array(protfeat), np.array(label), np.array(cond_set), np.array(group), name, seq
    else:
        return np.array(protfeat), np.array(label)
    
    
def extract_mut(path, cond, seq_length_min=None, feat_type='t5'):
    mutation_static_pos = []  # we do not consider mutations at the static positions
    protfeat, seq, cond_set, label, group, name = [], [], [], [], [], []
    data = np.load(path, allow_pickle=True)['data']
    
    if isinstance(cond, str):
        ## change this part ##
        range_temp = range(1, 61)
        range_conc = range(1, 201)
        range_nacl = range(1, 51)
        range_pH = range(61)
        range_book = {'temp': range_temp, 'conc': range_conc, 'nacl': range_nacl, 'pH': range_pH}
        base_cond = misc.normalize_condition(misc.base_cond_mutation)
        scale_value_cond = deepcopy(misc.scale_cond)
        bias_value_cond = deepcopy(misc.bias_cond)
        assert cond in range_book.keys(), "Invalid screen condition."
    
    for sample in tqdm(data, desc='Loading features from data'):
        sample_dict = edict(sample)
        if seq_length_min is not None:
            if len(sample_dict.seq) < seq_length_min: continue
        
        with h5py.File(sample_dict[f'{feat_type}_feature_path'], 'r') as f:
            fvalue = np.array(f['protein_feat'][0][:])
            flist = fvalue.tolist()
            
        if 'static_pos' in sample_dict:
            mutation_static_pos = sample_dict.static_pos
            
        if isinstance(cond, str):
            if sample_dict.conc != 0:
                scale_value_cond['conc'] = sample_dict.conc * misc.max_conc / 100
                
            for i in range_book[cond]:
                cond_ = deepcopy(base_cond)
                cond_[cond] = (i * scale_value_cond[cond] + bias_value_cond[cond]) / misc.max_values[cond]
                protfeat.append(flist)
                group.append(sample_dict.group)
                cond_set.append(list(cond_.values()))
                label.append(sample_dict.score)
                seq.append(sample_dict.seq)
                name.append(sample_dict.name)
                
        elif isinstance(cond, list):
            protfeat.append(flist)
            label.append(sample_dict.score)
            seq.append(sample_dict.seq)
            group.append(sample_dict.group)
            name.append(sample_dict.name)
            cond_set.append(cond)
        else:
            raise ValueError("Invalid condition input.")
        
    return np.array(protfeat), np.array(label), np.array(cond_set), np.array(group), name, seq, mutation_static_pos
    
    
def extract_screen(path, screen, seq_length_min=None, feat_type='t5'):
    protfeat, seq, cond_set, label, group, name = [], [], [], [], [], []
    
    ## change this part ##
    range_temp = range(1, 61)
    range_conc = range(1, 501)
    range_nacl = range(1, 51)
    range_pH = range(61)
    range_book = {'temp': range_temp, 'conc': range_conc, 'nacl': range_nacl, 'pH': range_pH}
    base_cond = misc.normalize_condition(misc.base_cond_screen)
    scale_value_cond = deepcopy(misc.scale_cond)
    bias_value_cond = deepcopy(misc.bias_cond)
    assert screen in range_book.keys(), "Invalid screen condition."
    
    data = np.load(path, allow_pickle=True)['data']
    for sample in tqdm(data, desc='Loading features from data'):
        sample_dict = edict(sample)
        if seq_length_min is not None:
            if len(sample_dict.seq) < seq_length_min: continue
        if sample_dict.group == 2: continue
        with h5py.File(sample_dict[f'{feat_type}_feature_path'], 'r') as f:
            fvalue = np.array(f['protein_feat'][0][:])
            flist = fvalue.tolist()
        if sample_dict.conc != 0:
            scale_value_cond['conc'] = sample_dict.conc * misc.max_conc / 100
        
        for i in range_book[screen]:
            cond = deepcopy(base_cond)
            cond[screen] = (i * scale_value_cond[screen] + bias_value_cond[screen]) / misc.max_values[screen]
            protfeat.append(flist)
            group.append(sample_dict.group)
            cond_set.append(list(cond.values()))
            label.append(sample_dict.score)
            seq.append(sample_dict.seq)
            name.append(sample_dict.name)
    return np.array(protfeat), np.array(label), np.array(cond_set), np.array(group), name, seq


def extract_screen_mul(path, screen, seq_length_min=None, feat_type='t5'):
    protfeat, seq, cond_set, label, group, name = [], [], [], [], [], []
    
    ## change this part ##
    range_temp = range(0, 61)
    range_conc = range(0, 51)
    range_nacl = range(1, 51)
    range_pH = range(61)
    range_book = {'temp': range_temp, 'conc': range_conc, 'nacl': range_nacl, 'pH': range_pH}
    base_cond = misc.normalize_condition(misc.base_cond_screen_multiple)
    scale_value_cond = deepcopy(misc.scale_cond)
    bias_value_cond = deepcopy(misc.bias_cond)
    
    # Load the data
    data = np.load(path, allow_pickle=True)['data']
    assert isinstance(screen, list), "The screen condition should be a list of two elements."
    assert len(screen) == 2, "The screen condition should be a list of two elements."
    screen_0, screen_1 = screen[0], screen[1]
    assert screen_0 in range_book.keys(), "Invalid screen_0 condition."
    assert screen_1 in range_book.keys(), "Invalid screen_1 condition."
    
    for sample in tqdm(data, desc='Loading features from data'):
        sample_dict = edict(sample)
        if seq_length_min is not None:
            if len(sample_dict.seq) < seq_length_min: continue
        if sample_dict.group == 2: continue
        with h5py.File(sample_dict[f'{feat_type}_feature_path'], 'r') as f:
            fvalue = np.array(f['protein_feat'][0][:])
            flist = fvalue.tolist()
        if sample_dict.conc != 0:
            scale_value_cond['conc'] = sample_dict.conc * misc.max_conc / 100
            
        for i in range_book[screen_0]:
            for j in range_book[screen_1]:
                cond = deepcopy(base_cond)
                cond[screen_0] = (i * scale_value_cond[screen_0] + bias_value_cond[screen_0]) / misc.max_values[screen_0]
                cond[screen_1] = (j * scale_value_cond[screen_1] + bias_value_cond[screen_1]) / misc.max_values[screen_1]
                protfeat.append(flist)
                group.append(sample_dict.group)
                cond_set.append(list(cond.values()))
                label.append(sample_dict.score)
                seq.append(sample_dict.seq)
                name.append(sample_dict.name)
    
    return np.array(protfeat), np.array(label), np.array(cond_set), np.array(group), name, seq


def cond_to_metric(cond):
    cond_out = np.zeros_like(cond)
    cond_out[:, 0] = cond[:, 0] * misc.max_temp
    cond_out[:, 1] = cond[:, 1] * misc.max_conc
    cond_out[:, 2] = cond[:, 2] * misc.max_pH
    cond_out[:, 3:9] = cond[:, 3:9] * misc.max_cagent
    cond_out[:, 9] = cond[:, 9] * misc.max_mgcl2
    cond_out[:, 10] = cond[:, 10] * misc.max_nacl
    cond_out[:, 11] = cond[:, 11] * misc.max_kcl
    cond_out[:, 12] = cond[:, 12] * misc.max_glyc
    return cond_out


def check_irregular(calc_results, screen):
    if screen == 'temp':
        search = 60
        protein_num = len(calc_results)//search
        found, value = [], []
        for i in range(protein_num):
            if calc_results[search*i+44] - calc_results[search*i+24] > 0.5:
                found.append(i)
                value.append(calc_results[search*i+44]-calc_results[search*i+24])
        if len(found) > 0:
            max_value = max(value)
            max_index = value.index(max_value)
            print(found, len(found), max_value, found[max_index])
        else:
            print("No irregular proteins found.")

    elif screen == 'nacl':
        search = 50
        protein_num = len(calc_results)//search
        found, value = [], []
        for i in range(protein_num):
            if calc_results[search*i+14] - calc_results[search*i+4] > 0.1:
                found.append(i)
                value.append(calc_results[search*i+14]-calc_results[search*i+4])
        if len(found) > 0:
            max_value = max(value)
            max_index = value.index(max_value)
            print(found, len(found), max_value, found[max_index])
        else:
            print("No irregular proteins found.")
            
    elif screen == 'pH':
        search = 41
        protein_num = len(calc_results)//search
        found_up, found_down = [], []
        for i in range(protein_num):
            if calc_results[search*i+40]-calc_results[search*i+20] > 0.1: # pH 9 - 7
                found_up.append(i)
            if calc_results[search*i]-calc_results[search*i+20] > 0.1: # pH 5 - 7
                found_down.append(i)
        if len(found_up) > 0.1 or len(found_down) > 0:
            print(found_up, len(found_up), found_down, len(found_down))
        else:
            print("No irregular proteins found.")
    
    
def moving_average(y, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd to ensure symmetry.")
    
    # Define the window: it ensures symmetry and smoothing.
    window = np.ones(int(window_size)) / float(window_size)
    y_padded = np.pad(y, (window_size//2, window_size//2), mode='edge')
    return np.convolve(y_padded, window, 'valid')


def savitzky_golay_smoothing(y, window_size, poly_order):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    if window_size < 1:
        raise ValueError("Window size must be greater than 1.")
    if window_size < poly_order + 2:
        raise ValueError("Window size must be at least the polynomial order + 2.")
    
    # Apply Savitzky-Golay filter
    return scipy.signal.savgol_filter(y, window_size, poly_order)


def draw_screen_graph(xvalue, prob, screen, filepath):
    
    fontscale = 1.0
    plt.rc('font', size=int(SMALL_SIZE * fontscale))          # controls default text sizes
    plt.rc('axes', titlesize=int(SMALL_SIZE * fontscale))     # fontsize of the axes title
    plt.rc('axes', labelsize=int(MEDIUM_SIZE * fontscale))    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=int(SMALL_SIZE * fontscale))    # fontsize of the tick labels
    plt.rc('ytick', labelsize=int(SMALL_SIZE * fontscale))    # fontsize of the tick labels
    # plt.rc('legend', fontsize=int(SMALL_SIZE * fontscale))    # legend fontsize
    plt.rc('figure', titlesize=int(BIGGER_SIZE * fontscale))  # fontsize of the figure title
    
    plt.figure(figsize=(20, 12)) 
    plt.plot(xvalue, prob, label=None, lw=3) 
    
    plt.axhline(
        y=0.5,
        color='r',
        linestyle='--',
        dashes=(10, 5),     # ← 점선 길이 10, 간격 5 로 시각적으로 넓게
        lw=2,               # ← 두께도 키움
        label=None
    )
    
    plt.ylim(0, 1)
    plt.xlabel(screen) 
    plt.ylabel("Probability") 
    # plt.title(f"Trend of LLPS probability with respect to {screen}")
    plt.legend() 
    plt.savefig(filepath, dpi=300)
    plt.close()

    
def draw_screen_graph_mul(xvalue0, xvalue1, prob, screen, filepath, filepath_excel=None, smooth=True):
    
    fontscale = 0.4
    plt.rc('font', size=int(SMALL_SIZE * fontscale))          # controls default text sizes
    plt.rc('axes', titlesize=int(SMALL_SIZE * fontscale))     # fontsize of the axes title
    plt.rc('axes', labelsize=int(MEDIUM_SIZE * fontscale))    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=int(SMALL_SIZE * fontscale))    # fontsize of the tick labels
    plt.rc('ytick', labelsize=int(SMALL_SIZE * fontscale))    # fontsize of the tick labels
    plt.rc('legend', fontsize=int(SMALL_SIZE * fontscale))    # legend fontsize
    plt.rc('figure', titlesize=int(BIGGER_SIZE * fontscale))  # fontsize of the figure title
    
    # Prepare grid data for surface plot
    xvalue_grid, yvalue_grid = list(np.unique(xvalue0)), list(np.unique(xvalue1))
    grid_x = xvalue0.reshape(len(xvalue_grid), len(yvalue_grid))
    grid_y = xvalue1.reshape(len(xvalue_grid), len(yvalue_grid))
    grid_z = prob.reshape(len(xvalue_grid), len(yvalue_grid))

    if smooth:
        # Apply Gaussian smoothing
        grid_z = gaussian_filter(grid_z, sigma=3.0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
    ax.set_xlabel(screen[0])
    ax.set_ylabel(screen[1])
    ax.set_zlabel('Probability', labelpad=2)
    ax.set_zlim(0, 1)
    
    if grid_z[-1, -1] >= grid_z[0, 0]:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    
    plt.tight_layout()
    plt.savefig(filepath)
    
    if filepath_excel is not None:
        # Add contour at prob = 0.5
        contour = ax.contour(grid_x, grid_y, grid_z, levels=[0.5], colors='red', linestyles='solid')

        # Extract contour data
        for c in contour.collections:
            for path in c.get_paths():
                v = path.vertices
                x_contour = v[:, 0]
                y_contour = v[:, 1]
                df_results = pd.DataFrame({screen[0]: x_contour, screen[1]: y_contour})
                df_results.to_excel(filepath_excel, index=False)
    
    plt.close()
    
    
def draw_mut_graph(delta_high, delta_low, filepath):
    seq_len = len(delta_high)
    assert len(delta_low) == seq_len, "Lengths of delta_high and delta_low should be the same."
    
    fontscale = 1.0
    plt.rc('font', size=int(SMALL_SIZE * fontscale))          # controls default text sizes
    plt.rc('axes', titlesize=int(SMALL_SIZE * fontscale))     # fontsize of the axes title
    plt.rc('axes', labelsize=int(MEDIUM_SIZE * fontscale))    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=int(SMALL_SIZE * fontscale))    # fontsize of the tick labels
    plt.rc('ytick', labelsize=int(SMALL_SIZE * fontscale))    # fontsize of the tick labels
    plt.rc('legend', fontsize=int(SMALL_SIZE * fontscale))    # legend fontsize
    plt.rc('figure', titlesize=int(BIGGER_SIZE * fontscale))  # fontsize of the figure title
    
    plt.figure(figsize=(20, 10))
    plt.fill_between(range(seq_len), delta_high, delta_low, color='red', alpha=0.3, label='Delta Probability Range')
    plt.axhline(0, color='black', linewidth=1)

    # Adding labels and title
    plt.xlabel('Position')
    plt.ylabel('Delta Probability')
    # plt.title('Shaded Line Graph of Score Variations')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    
    
def save_mutated_protein(original_filepath, mutated_protein, static_pos):
    if '_mut' in original_filepath:
        splitted = original_filepath.split('_mut')
        splitted[-1] = str(int(splitted[-1][:-4]) + 1) +'.xlsx'
        output_filepath = '_mut'.join(splitted)
    else:
        output_filepath = original_filepath[:-4] + '_mut1.xlsx'
        
    data = np.load(original_filepath, allow_pickle=True)['data']
    static_pos = [str(pos) for pos in static_pos]
    
    df_results = pd.DataFrame({'Protein name': [data[0].name],
                            'Sequence': [mutated_protein],
                            'Conc': [data[0].conc * misc.max_conc],
                            'pH': [data[0].pH * misc.max_pH],
                            'temp': [data[0].temp * misc.max_temp],
                            'MgCl2': [data[0].salt[0] * misc.max_mgcl2],
                            'NaCl': [data[0].salt[1] * misc.max_nacl],
                            'KCl': [data[0].salt[2] * misc.max_kcl],
                            'PEG500': [data[0].cagent[0] * misc.max_cagent],
                            'PEG3350': [data[0].cagent[1] * misc.max_cagent],
                            'PEG8000': [data[0].cagent[2] * misc.max_cagent],
                            'Ficoll 400': [data[0].cagent[3] * misc.max_cagent],
                            'Dextran 40': [data[0].cagent[4] * misc.max_cagent],
                            'Dextran 70': [data[0].cagent[5] * misc.max_cagent],
                            'Glycerol': [data[0].glyc * misc.max_glyc],
                            'label': [data[0].score],
                            'group': [data[0].group],
                            'Mutation Pos': [':' + str(','.join(static_pos))]
                            })
    
    df_results.to_excel(output_filepath.replace('processed', 'raw'), index=False, sheet_name='Sheet1')
    
    
def draw_cross_roc_curve(mean_fpr, tprs, aucs, subplots, filepath):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.rcParams.update({'font.size': 14})
    for i, subplot in enumerate(subplots):
        subplot.plot(ax=ax, lw=1, alpha=0.3, plot_chance_level=(i == len(subplots) - 1))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=None,
    )
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    handles, labels = ax.get_legend_handles_labels()
    labels = [
        re.sub(r'(Fold )(\d+)',
               lambda m: f"{m.group(1)}{int(m.group(2)) + 1}", lab)
        for lab in labels
    ]
    
    # 'Chance level'만 제거
    filtered = [
        (h, l) for h, l in zip(handles, labels)
        if "Chance level" not in l
    ]
    if filtered:
        handles, labels = zip(*filtered)

    ax.legend(handles, labels, loc="lower right", fontsize=12)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    
def draw_cross_pr_curve(mean_recall, precision, auprcs, subplots, filepath):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.rcParams.update({'font.size': 14})
    for i, subplot in enumerate(subplots):
        subplot.plot(ax=ax, lw=1, alpha=0.3, label='Fold %d (AUC = %0.2f)' % (i + 1, auprcs[i]))
    
    mean_precision = np.mean(precision, axis=0)
    mean_precision[0] = 1.0
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)
    ax.plot(
        mean_recall,
        mean_precision,
        color="b",
        label=r"Mean (AUC = %0.2f $\pm$ %0.2f)" % (mean_auprc, std_auprc),
        lw=2,
        alpha=0.8,
    )

    std_precision = np.std(precision, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    ax.fill_between(
        mean_recall,
        precision_lower,
        precision_upper,
        color="grey",
        alpha=0.2,
        label=None,
    )
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.legend(loc="lower left", fontsize=12)
    plt.savefig(filepath, dpi=300)
    plt.close()
