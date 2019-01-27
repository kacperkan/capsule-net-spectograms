import re
from pathlib import Path

import pandas as pd
import numpy as np

submissions_regex = re.compile(r".*_(\d+).csv")
debug = False


def dummy_data(frame: pd.DataFrame) -> pd.DataFrame:
    frame['c1'] = frame['class']
    frame['c2'] = frame['class'] + 1
    frame['c3'] = frame['class'] + 2
    return frame


def add_vote_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame[0] = pd.Series(np.zeros(len(frame)), index=frame.index)
    frame[1] = pd.Series(np.zeros(len(frame)), index=frame.index)
    frame[2] = pd.Series(np.zeros(len(frame)), index=frame.index)
    return frame


def combine(folder: str):
    folder = Path(folder)
    all_files = folder.rglob("*.csv")
    necessary_files = [file for file in all_files if submissions_regex.match(file.as_posix())]
    necessary_files = [
        file.parent / ('_'.join(file.name.split('_')[:-1]) + '.csv') for file in necessary_files
    ]
    base = pd.read_csv(necessary_files[0], index_col='id')
    base = add_vote_columns(base)
    for f in necessary_files[1:]:
        additional = pd.read_csv(f, index_col='id')

        base.loc[additional['class'] == 0, 0] += 1
        base.loc[additional['class'] == 1, 1] += 1
        base.loc[additional['class'] == 2, 2] += 1
        # base['c1'] += additional['c1'] / len(necessary_files)
        # base['c2'] += additional['c2'] / len(necessary_files)
        # base['c3'] += additional['c3'] / len(necessary_files)

    # base['class'] = base[['c1', 'c2', 'c3']].values.argmax(axis=1)
    # base.drop(['c1', 'c2', 'c3'], axis=1, inplace=True)
    base['class'] = base[[0, 1, 2]].values.argmax(axis=1)
    base.drop([0, 1, 2], axis=1, inplace=True)
    base.to_csv((folder / '{}.csv'.format(folder.name)).as_posix())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("folder")
    args = parser.parse_args()

    combine(args.folder)
