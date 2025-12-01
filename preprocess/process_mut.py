
import argparse
import preprocess.misc as misc
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
import random
from copy import deepcopy


#### setting ####
mul_range = 10.0  # [0.1x ~ 10x]
# [max value, sampled_num, sampled_min, sampled_max, unit of sample (if not specified, 0)]
guide_all = {'temp': [misc.max_temp, 50, 0, misc.max_temp, 1], 
            'conc': [misc.max_conc, 10, 0, misc.max_conc, 0], 
            'pH': [misc.max_pH, 10, 0, misc.max_pH, 0.1],
            'mgcl2': [misc.max_mgcl2, 10, 0, misc.max_mgcl2, 1],
            'nacl': [misc.max_nacl, 50, 0, misc.max_nacl, 10],
            'kcl': [misc.max_kcl, 50, 0, misc.max_kcl, 10],
            'cagent': [misc.max_cagent, 30, 0, misc.max_cagent, 1],
            'glyc': [misc.max_glyc, 10, 0, misc.max_glyc, 1]}


# decode fasta file
def decode_fasta(fasta_info):
    splitted = fasta_info.split('\n')
    if '>' in splitted[0]:
        if len(splitted[0].split('|')) > 1:
            entry_id = splitted[0].split('|')[1]
            if ('length' in entry_id) or ('linker' in entry_id): 
                entry_id = None
        else: entry_id = None
        seq = ''.join([seq_part for seq_part in splitted[1:]])
    else:
        entry_id = None
        seq = ''.join([seq_part for seq_part in splitted[0:]])
    return entry_id, seq

def get_range_data(minvalue, maxvalue, num_split, num_unit, reverse_order=False, range_limit=False):
    if num_unit == 0:
        bias = (maxvalue - minvalue) / num_split
        return np.linspace(minvalue, maxvalue - bias, num_split)
    else:
        values = np.arange(minvalue, maxvalue - num_unit, num_unit)
        if len(values) > num_split and range_limit:
            if reverse_order:
                return values[-num_split:]
            return values[:num_split]
        elif not range_limit and len(values) < 10:
            bias = (maxvalue - minvalue) / 10
            return np.linspace(minvalue, maxvalue - bias, 10)
        else:
            return values

# process ~ data
def process_data(data, tag):
    if isinstance(data, (float, int)):
        return [float(data) / guide_all[tag][0]]
    else:
        if tag == 'temp':
            data = data.replace('RT', '25')
        if '~' in data:  # data with range
            splitted = data.split('~')
            assert(len(splitted) == 2)
            # case 1 : a ~ b
            if not '' in splitted:
                reverse_order, range_limit = False, False
                minvalue, maxvalue = float(splitted[0]), float(splitted[1])
            # case 2 : a ~
            elif splitted[1] == '':
                reverse_order, range_limit = False, True
                minvalue, maxvalue = float(splitted[0]), min(guide_all[tag][3], float(splitted[0]) * mul_range)
            # case 3 : ~ b
            elif splitted[0] == '':
                reverse_order, range_limit = True, True
                minvalue, maxvalue = max(guide_all[tag][2], float(splitted[1]) / mul_range), float(splitted[1])
            else:
                raise NotImplementedError('Not implemented case: {}'.format(data))
            
            num_split, num_unit = guide_all[tag][1], guide_all[tag][4]
            output = get_range_data(minvalue, maxvalue, num_split, num_unit, reverse_order, range_limit)
            output = output / guide_all[tag][0]
            return output.tolist()
        else:
            return [float(data) / guide_all[tag][0]]



def main(cfg):
    
    # read given excel data
    filepath = Path(f'data/raw/{cfg.srcfilename}')
    data_array = pd.read_excel(str(filepath), engine='openpyxl').to_numpy()
    protein_info = []
    
    seq_idx = 0
    for i in range(len(data_array)):
        data_value = data_array[i]
        
        # basic information
        name = data_value[0]
        entry_id, seq = decode_fasta(data_value[1])
        
        # process data
        conc = process_data(data_value[2], 'conc')
        pH = process_data(data_value[3], 'pH')
        temp = process_data(data_value[4], 'temp')
        mgcl2 = process_data(data_value[5], 'mgcl2')
        nacl = process_data(data_value[6], 'nacl')
        kcl = process_data(data_value[7], 'kcl')
        glyc = process_data(data_value[14], 'glyc')
        processed_data = [conc, pH, temp, mgcl2, nacl, kcl, glyc]
        
        # read cagent data
        for n in range(8, 14):
            processed_data += [process_data(data_value[n], 'cagent')]
        
        processed_data += [[data_value[15]]]
        processed_data += [[data_value[16]]]
        if len(data_value) > 17:
            # find mutated sequence and define static position
            mut_signs = data_value[17][1:].split(',')
            static_pos = [int(mut_sign) for mut_sign in mut_signs]
        else:
            static_pos = []
        
        # combine all the cases together
        processed_data = list(itertools.product(*processed_data))
        
        if len(processed_data) > 100:
            cutoff = max(100, len(processed_data) // 10)
            processed_data = random.sample(processed_data, cutoff)
    
        # get information
        for data in processed_data:
            # register protein
            protein_dict = edict()
            protein_dict.name = name
            protein_dict.entry_id = entry_id
            protein_dict.seq = seq
            protein_dict.seq_id = seq_idx
            protein_dict.static_pos = static_pos
            
            # get processed information (order aligned with other dataset)
            protein_dict.conc = data[0]
            protein_dict.pH = data[1]
            protein_dict.temp = data[2]
            protein_dict.cagent = np.array(data[7:13])
            protein_dict.salt = np.array(data[3:6])
            protein_dict.glyc = data[6]
            
            # assign score
            protein_dict.score = data[13]
            protein_dict.group = i
            
            # save data
            protein_info.append(protein_dict)
            
            mut_num = 0
            for mut in range(len(seq)):
                for aa in misc.amino_acids:
                    if aa == seq[mut]: continue
                    protein_dict_mut = deepcopy(protein_dict)
                    
                    protein_dict_mut.name = name + '_' + seq[mut] + str(mut+1) + aa
                    protein_dict_mut.seq = seq[:mut] + aa + seq[mut+1:]
                    protein_dict_mut.seq_id = seq_idx + mut_num + 1
                    
                    # save data
                    protein_info.append(protein_dict_mut)
                    mut_num += 1
        seq_idx += (mut_num + 1)
            
            
    # save as npz file
    output_path = Path(f'data/processed/{filepath.stem}')
    np.savez_compressed(output_path, data=protein_info)
    
    
if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--srcfilename', type=str, default='pspire_train.npz', help='source filename')
    args = parser.parse_args()
    main(args)