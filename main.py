from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import yaml

from matrix.signal_matrix_factory import SignalMatrixFactory
from matrix.PIP_loss_calculator import MonteCarloEstimator
from utils.tokenizer import SimpleTokenizer
from utils.reader import ReaderFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='construct pmi matrix from corpus')
    parser.add_argument('--algorithm', required=True, type=str, help='embedding algorithm')
    parser.add_argument('--file', required=True, type=str, help='corpus_file')
    parser.add_argument('--config_file', required=False, type=str, help='config file for the algorithm containing parameter settings')
    args = parser.parse_args()
   
    config_file = args.config_file

    with open(args.config_file, "r") as f:
        cfg = yaml.load(f)

    reader = ReaderFactory.produce(args.file[-3:])
    data = reader.read_data(args.file)
    tokenizer = SimpleTokenizer()
    indexed_corpus = tokenizer.do_index_data(data,
            n_words=cfg.get('vocabulary_size'),
            min_count=cfg.get('min_count'))
    factory = SignalMatrixFactory(indexed_corpus)

    signal_matrix = factory.produce(args.algorithm.lower())
    path = signal_matrix.param_dir
    signal_matrix.inject_params(cfg)
    signal_matrix.estimate_signal()
    signal_matrix.estimate_noise()
    signal_matrix.export_estimates()

    pip_calculator = MonteCarloEstimator()
    pip_calculator.get_param_file(path, "estimates.yml")
    pip_calculator.estimate_signal()
    pip_calculator.estimate_pip_loss()
    pip_calculator.plot_pip_loss()
