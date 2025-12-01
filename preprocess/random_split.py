
import argparse
import numpy as np
from pathlib import Path
import pandas as pd


def check_validity(data):
    amino_acids = ['O', 'X', 'B', 'Z', 'J']
    indexes = []
    for i, sample in enumerate(data):
        mask = any([char in amino_acids for char in sample['seq']])
        if mask: indexes.append(i)
    return indexes


def main(args):
    
    srcfilename = args.srcfilename
    train_ratio = args.train_ratio
    np.random.seed(args.seed)
    
    # setting
    root_folder = Path('data')
    input_path = root_folder / 'processed' / srcfilename
    
    # read data
    data = np.load(input_path, allow_pickle=True)['data']
    idx = check_validity(data)
    data = np.delete(data, idx)
    data = list(data)
    
    dict_data = {}
    for datum in data:
        for key in datum.keys():
            if key not in dict_data:
                dict_data[key] = []
            dict_data[key].append(datum[key])
            
    df = pd.DataFrame(dict_data)
        
    # Unique groups and their sizes
    groups = df['group'].unique()
    group_sizes = df.groupby('group').size()

    # Shuffle groups to ensure random distribution
    shuffled_groups = np.random.permutation(groups)

    # Calculate the target training size
    target_train_size = int(len(df) * train_ratio)

    train_groups = []
    test_groups = []
    current_train_size = 0

    # Iterate through the shuffled groups, adding them to the training set until the target size is reached
    for group in shuffled_groups:
        if current_train_size + group_sizes[group] <= target_train_size:
            train_groups.append(group)
            current_train_size += group_sizes[group]
        else:
            test_groups.append(group)


    # Split the data based on the groups
    train_df = df[df['group'].isin(train_groups)]
    test_df = df[df['group'].isin(test_groups)]

    # Output sizes for verification
    print(f"Total data size: {len(data)}")
    print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")
    
    # save as npz file
    output_path = root_folder / 'processed' / f'{input_path.stem}_train'
    np.savez_compressed(output_path, data=train_df.to_dict('records'))
    output_path = root_folder / 'processed' / f'{input_path.stem}_test'
    np.savez_compressed(output_path, data=test_df.to_dict('records'))
    
    

if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--srcfilename', type=str, default='pspire_train.npz', help='source filename')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='source filename')
    parser.add_argument('--seed', type=int, default=42, help='source filename')
    args = parser.parse_args()
    main(args)