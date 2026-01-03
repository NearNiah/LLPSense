

import math
import torch
import numpy as np
import pandas as pd
from einops import repeat
from scipy import signal
from iupred.iupred2a_lib import iupred
from easydict import EasyDict as edict
from Bio.SeqUtils.ProtParam import ProteinAnalysis



class EngFeat:
    
    def __init__(self, name, n_window=20, cutoff=7, device='cpu'):
        self.name = name
        self.device = device
        
        # define amino acids and their hydrophobicity index
        self.RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        # Kyte & Doolittle {kd} index of hydrophobicity
        self.HP = {'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
                   'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
                   'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
                   'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2, 'U': 0.0}
        
        # variable for low complexity score
        self.n_window = n_window
        self.cutoff = cutoff
        
    def calc_idrpred(self, seq):
        glob = iupred(str(seq), 'glob')
        short = iupred(str(seq), 'short')
        long = iupred(str(seq), 'long')
        idrpred = edict(glob=glob, short=short, long=long)
        return idrpred
    
    def convolve_signal(self, sig, window=25):
        win = signal.windows.hann(window)
        sig = signal.convolve(sig, win, mode='same') / sum(win)
        return sig
    
    def hydrophobicity(self, seqanalysis):
        HB = 0
        for k in range(0, len(self.RESIDUES)):
            HB = HB + seqanalysis.count_amino_acids()[self.RESIDUES[k]] * self.HP[self.RESIDUES[k]]        
        
        return HB

    def Shannon_entropy(self, seqanalysis):
        entropy = 0
        for k in range(0, len(self.RESIDUES)):
            if seqanalysis.get_amino_acids_percent()[self.RESIDUES[k]] == 0:
                entropy = entropy + 0
            else:
                entropy = entropy - math.log2(seqanalysis.get_amino_acids_percent()[self.RESIDUES[k]]) * seqanalysis.get_amino_acids_percent()[self.RESIDUES[k]]        
        return entropy
    
    @torch.no_grad()
    def get_embedding(self, seqs, tdevice):
        length_seq = max([len(seq) for seq in seqs])
        embedding_repr = []
        for seq in seqs:
            protein = edict()
            protein.seq = seq
            protein.feat_glob = edict()
            
            # amino acid analysis
            for res in self.RESIDUES:
                protein.feat_glob['fraction_'+res] = protein.seq.count(res) / len(protein.seq)
            protein.feat_glob.length = len(protein.seq)
            seqanalysis = ProteinAnalysis(protein.seq)
            protein.feat_glob.IEP = seqanalysis.isoelectric_point()
            protein.feat_glob.molecular_weight = seqanalysis.molecular_weight()
            if 'U' not in protein.seq:
                protein.feat_glob.gravy = seqanalysis.gravy()
                
            # calculate idr given seq
            idr = self.calc_idrpred(protein.seq)
            # protein.feat_glob.iupred = idr
            idr_glob = idr.glob[0]
            protein.feat_glob.idr_percetage = sum(i > .5 for i in list(idr_glob))
            protein.feat_glob.idr_50 = sum(i > .5 for i in list(idr_glob)) / len(str(protein.seq))
            protein.feat_glob.idr_60 = sum(i > .6 for i in list(idr_glob)) / len(str(protein.seq))
            protein.feat_glob.idr_70 = sum(i > .7 for i in list(idr_glob)) / len(str(protein.seq))
            protein.feat_glob.idr_80 = sum(i > .8 for i in list(idr_glob)) / len(str(protein.seq))
            protein.feat_glob.idr_90 = sum(i > .9 for i in list(idr_glob)) / len(str(protein.seq))
            
            # hydrophobic analysis (number of hydrophobic residues)
            hpilist = pd.Series(list(protein.seq)).map(self.HP).tolist()
            # protein.feat_glob.HydroPhobicIndex = hpilist
            sw = self.convolve_signal(hpilist, window=30)
            protein.feat_glob.hpi0 = sum(i < -1.5 for i in sw) / len(sw)  # hpi_<-1.5_frac
            protein.feat_glob.hpi1 = sum(i < -2.0 for i in sw) / len(sw)  # hpi_<-2.0_frac
            protein.feat_glob.hpi2 = sum(i < -2.5 for i in sw) / len(sw)  # hpi_<-2.5_frac
            protein.feat_glob.hpi3 = sum(i < -1.5 for i in sw)  # hpi_<-1.5
            protein.feat_glob.hpi4 = sum(i < -2.0 for i in sw)  # hpi_<-2.0
            protein.feat_glob.hpi5 = sum(i < -2.5 for i in sw)  # hpi_<-2.5
            
            # get shannon entropy  # not envolved in 55 features
            protein.feat_glob.shanon_entropy = self.Shannon_entropy(seqanalysis)
            
            # biochemical_combinations
            protein.feat_glob.Asx = protein.feat_glob.fraction_D + protein.feat_glob.fraction_N
            protein.feat_glob.Glx = protein.feat_glob.fraction_E + protein.feat_glob.fraction_Q
            protein.feat_glob.Xle = protein.feat_glob.fraction_I + protein.feat_glob.fraction_L
            protein.feat_glob.Pos_charge = protein.feat_glob.fraction_K + protein.feat_glob.fraction_R + protein.feat_glob.fraction_H
            protein.feat_glob.Neg_charge = protein.feat_glob.fraction_D + protein.feat_glob.fraction_E
            protein.feat_glob.Aromatic = protein.feat_glob.fraction_F + protein.feat_glob.fraction_W + protein.feat_glob.fraction_Y + protein.feat_glob.fraction_H
            protein.feat_glob.Alipatic = protein.feat_glob.fraction_V + protein.feat_glob.fraction_I + protein.feat_glob.fraction_L + protein.feat_glob.fraction_M
            protein.feat_glob.Small = protein.feat_glob.fraction_P + protein.feat_glob.fraction_G + protein.feat_glob.fraction_A + protein.feat_glob.fraction_S
            protein.feat_glob.Hydrophilic = protein.feat_glob.fraction_S + protein.feat_glob.fraction_T + protein.feat_glob.fraction_H + \
                                    protein.feat_glob.fraction_N + protein.feat_glob.fraction_Q + protein.feat_glob.fraction_E + \
                                    protein.feat_glob.fraction_D + protein.feat_glob.fraction_K + protein.feat_glob.fraction_R
            protein.feat_glob.Hydrophobic = protein.feat_glob.fraction_V + protein.feat_glob.fraction_I + protein.feat_glob.fraction_L + \
                                    protein.feat_glob.fraction_F + protein.feat_glob.fraction_W + protein.feat_glob.fraction_Y + \
                                    protein.feat_glob.fraction_M
            
            protein.feat_glob.alpha_helix = protein.feat_glob.fraction_V + protein.feat_glob.fraction_I + protein.feat_glob.fraction_Y + \
                                    protein.feat_glob.fraction_F + protein.feat_glob.fraction_W + protein.feat_glob.fraction_L
            protein.feat_glob.beta_turn = protein.feat_glob.fraction_N + protein.feat_glob.fraction_P + protein.feat_glob.fraction_G + protein.feat_glob.fraction_S
            protein.feat_glob.beta_sheet = protein.feat_glob.fraction_E + protein.feat_glob.fraction_M + protein.feat_glob.fraction_A + protein.feat_glob.fraction_L
            # Calculates the aromaticity value of a protein according to Lobry, 1994. 
            # It is simply the relative frequency of Phe+Trp+Tyr.
            protein.feat_glob.aromaticity = protein.feat_glob.fraction_F + protein.feat_glob.fraction_W + protein.feat_glob.fraction_Y
            
            # Determine low complexity scores
            n_halfwindow = self.n_window // 2
            lcs_acids, sig = list(), list()
            lc_bool = np.zeros(len(protein.seq), dtype=np.bool_)
            for i in range(len(protein.seq)):
                
                # calculate complexity
                if i < n_halfwindow:
                    peptide = protein.seq[:self.n_window]        
                elif i + n_halfwindow > len(protein.seq):
                    peptide = protein.seq[-self.n_window:]        
                else:
                    peptide = protein.seq[i - n_halfwindow:i + n_halfwindow]
                complexity = len(set(peptide))
                
                # determine mask
                low_bound = max(0, i - n_halfwindow)
                high_bound = min(len(protein.seq) - 1, i + n_halfwindow)
                if complexity <= self.cutoff:
                    for bool_index in (low_bound, high_bound):
                        lc_bool[bool_index] = True
                    lcs_acids.append(protein.seq[i])
                sig.append(complexity)
            
            # Adding low complexity scores to list
            low_complexity_list = pd.DataFrame({'bool': lc_bool, 'acid': list(protein.seq)}, index=None)
            protein.feat_glob.lcs_lowest_complexity = min(sig)
            protein.feat_glob.lcs_score = len(low_complexity_list.loc[low_complexity_list['bool'] == True])
            protein.feat_glob.lcs_fraction = len(low_complexity_list.loc[low_complexity_list['bool'] == True]) / len(protein.seq)
            
            global_feature = []
            for key, feat in protein.feat_glob.items():
                global_feature.append(feat)
            global_feature = np.array(global_feature)
            embedding_repr.append(torch.tensor(global_feature).to(tdevice))
        embedding_repr = torch.stack(embedding_repr, dim=0)
        embedding_repr = repeat(embedding_repr, 'b c -> b l c', l=length_seq)
        return embedding_repr