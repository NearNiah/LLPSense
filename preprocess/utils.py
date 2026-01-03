import os
import h5py
import shutil
import pandas as pd


def create_dir(path, exist_ok=True):
    if not exist_ok and path.exists():
        shutil.rmtree(str(path))
    os.makedirs(str(path), exist_ok=True)
    return path


def read_h5py(filepath, tag):
    assert os.path.exists(filepath), f"Filepath {filepath} does not exist."
    with h5py.File(filepath, 'r') as f:
        data = f[tag][:]
    return data


def write_h5py(filepath, data, tag):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset(tag, data=data, dtype='f', compression="gzip")
    return filepath


def get_data(file_path, col_axis=[]):
    df = pd.read_excel(file_path, engine='openpyxl')
    num_rows = len(df.index)
    output = dict()
    for n in col_axis: output[n] = [df.columns[n]]
    for i in range(num_rows):
        for n in col_axis:
            output[n].append(df.at[i, df.columns[n]])
    return output
