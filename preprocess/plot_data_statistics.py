
import argparse
import numpy as np
from pathlib import Path
import preprocess.misc as misc
import plotly.graph_objects as go
from preprocess.utils import create_dir



def main(args):
    
    srcfilename = args.srcfilename
    root_folder = Path('data')
    input_path = root_folder / 'processed' / srcfilename
    data = np.load(str(input_path), allow_pickle=True)['data']
    conditions = {'temp': [], 'conc': [], 'pH': [], 'score': []}
    titles = {'temp': 'Temperature (°C)', 'conc': 'Concentration (\u00B5M)', 'pH': 'pH'}
    maxvalue = {'temp': misc.max_temp, 'conc': misc.max_conc, 'pH': misc.max_pH, 'score': 1.0}

    for sample in data:
        for key in conditions.keys():
            conditions[key].append(sample[key])
            

    plot_number = 30
    for key in conditions.keys():
        if key == 'score': continue
        
        data_0 = [sample[key] * maxvalue[key] for sample in data if sample['score'] == 0]
        score_0 = [0 for sample in data if sample['score'] == 0]
        
        data_1 = [sample[key] * maxvalue[key] for sample in data if sample['score'] == 1]
        score_1 = [1 for sample in data if sample['score'] == 1]
        
        # Create a bar chart
        interval = maxvalue[key] / plot_number
        fig = go.Figure(data=[
            go.Histogram(name='Non LLPS', x=data_0, xbins=dict(start=0, size=interval, end=maxvalue[key]), marker=dict(color='rgb(255, 0, 0)')),
            go.Histogram(name='LLPS', x=data_1, xbins=dict(start=0, size=interval, end=maxvalue[key]), marker=dict(color='rgb(0, 0, 255)')),
        ])

        # Update the layout
        fig.update_layout(
            barmode='group',
            xaxis_title=titles[key],
            yaxis_title='Counts',
            plot_bgcolor='rgba(255, 255, 255, 0.0)'
        )
        fig.write_image(str(root_folder / f'{input_path.stem}_{key}.png'))
        
        
        
if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--srcfilename', type=str, default='pspire_train.npz', help='source filename')
    args = parser.parse_args()
    main(args)