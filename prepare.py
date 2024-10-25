import os, sys
import glob
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

def find_file(filename, search_path):
    path = Path(search_path)
    for file in path.rglob(filename):
        return file

def main(args):

    data={'path':[], 'speaker':[], 'intelligibility': [] }
    for csv in args.input_csv:
        df = pd.read_csv(csv)
        for idx, row in df.iterrows():
            intelligibility = row['intelligiblity']
            key = row['key']
            speaker = key[:-3]
            filename = key + ".wav"
            path = find_file(filename, args.search_dir)

            data['path'].append(os.path.abspath(path))
            data['intelligibility'].append(intelligibility)
            data['speaker'].append(speaker)
   
    df = pd.DataFrame.from_dict(data, orient='columns')
    df.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', nargs='*', required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--search_dir', type=str, required=True)

    args=parser.parse_args()
    main(args)
