import importlib
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from preprocess.utils import create_dir, write_h5py
from easydict import EasyDict as edict


def check_validity(data):
    amino_acids = ['O', 'X', 'B', 'Z', 'J']
    indexes = []
    for i, sample in enumerate(data):
        mask = any([char in amino_acids for char in sample['seq']])
        if mask: indexes.append(i)
    return indexes


def main(args):

    device = args.device
    feature_type = args.feature_type
    srcfilename = args.srcfilename

    # setting
    root_folder = Path('data')
    output_folder = create_dir(root_folder / feature_type)
    input_path = root_folder / 'processed' / srcfilename
    output_path = create_dir(output_folder / input_path.stem)
    assert input_path.exists(), f'input file not found: {input_path}'
    
    # load feature extractor
    if feature_type == 't5':
        module = importlib.import_module(f'preprocess.embedding.t5')
        class_name = 'ProtT5Embedder'
    elif feature_type == 't5_seq':
        module = importlib.import_module(f'preprocess.embedding.t5_seq')
        class_name = 'ProtT5Embedder'
    elif feature_type == 'eng':
        module = importlib.import_module(f'preprocess.embedding.eng')
        class_name = 'EngFeat'
    elif feature_type == 'esm':
        module = importlib.import_module(f'preprocess.embedding.esm')
        class_name = 'ESMEmbedder'
    else:
        raise NotImplementedError(f'feature type not found: {feature_type}')
    model = getattr(module, class_name)(name=feature_type, device=device)

    # extract feature
    data = np.load(input_path, allow_pickle=True)['data']
    idx = check_validity(data)
    data = np.delete(data, idx)
    
    # save feature
    for idx, data_part in enumerate(tqdm(data)):
        protein = edict(data_part)
        feature_path = output_path / f'{protein.seq_id:06}.hdf5'
        if not feature_path.exists():
            protein_feat = model.get_embedding([protein.seq], tdevice='cpu')[0]
            write_h5py(str(feature_path), protein_feat.numpy(), 'protein_feat')
        data[idx][f'{feature_type}_feature_path'] = str(feature_path)
    
    # overwrite as npz file
    np.savez_compressed(input_path, data=data)
    
    
if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--feature_type', type=str, default='t5', help='feature to extract')
    parser.add_argument('--srcfilename', type=str, default='pspire_train.npz', help='source filename')
    args = parser.parse_args()
    main(args)