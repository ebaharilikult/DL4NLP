import os
import sys

import pandas as pd
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from scipy import sparse
from scipy.sparse import csr_matrix


def transform_features(input_path: str, features_output_path: str, labels_output_path: str):
    file_name: str = input_path
    cols: Index = pd.read_csv(file_name, compression='zip',
                              sep='\t', nrows=1).columns

    train_features: csr_matrix = None
    train_labels: Series = None

    # read features
    for df in pd.read_csv(file_name, compression='zip', sep='\t', chunksize=3000, dtype='float16', usecols=cols[1::]):
        sp = sparse.csr_matrix(df.values)
        if train_features is None:
            train_features = sp
        else:
            train_features = sparse.vstack(
                (train_features, sp), format='csr')

    # save features
    sparse.save_npz(features_output_path, train_features, compressed=True)

    # read labels
    labels = pd.read_csv(file_name, compression='zip', sep='\t',
                         usecols=[0], index_col=False, encoding='ISO-8859-1')
    train_labels = labels[labels.columns[0]]

    # save labels
    train_labels.to_pickle(labels_output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the base directory of the features to transfom.\nusuage: featureTransformer.py <baseDir>")
    else:
        dir = sys.argv[1]
        print("Using dir:" + dir)
        transform_features(dir + '/train_features.zip', dir +
                           '/train_features.npz', dir + '/train_labels.pkl')
        transform_features(dir + '/eval_features.zip', dir +
                           '/eval_features.npz', dir + '/eval_labels.pkl')
        transform_features(dir + '/test_features.zip', dir +
                           '/test_features.npz', dir + '/test_labels.pkl')
