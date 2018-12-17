from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp

from six.moves import xrange  # pylint: disable=redefined-builtin


class SignalMatrix():
    def __init__(self, corpus=None):
        self.corpus = corpus
        self._param_dir = "params/{}".format(self.__class__.__name__)
        sp.check_output("mkdir -p {}".format(self._param_dir), shell=True)
        self._get_vocab_size()

    def _get_vocab_size(self):
        """ words are {0, 1, ..., n_words - 1}"""
        vocabulary_size = 1
        for idx, center_word_id in enumerate(self.corpus):
            if center_word_id + 1> vocabulary_size:
                vocabulary_size = center_word_id + 1
        self.vocabulary_size = vocabulary_size
        print("vocabulary_size={}".format(self.vocabulary_size))
    
    @property
    def param_dir(self):
        return self._param_dir

    def estimate_signal(self, enable_plot=False):
        matrix = self.construct_matrix(self.corpus)
        self.matrix = matrix
        U, D, V = np.linalg.svd(matrix)
        if enable_plot:
            plt.plot(D)
            plt.savefig('{}/sv.pdf'.format(self._param_dir))
            plt.close()
        self.spectrum = D
        with open("{}/sv.pkl".format(self._param_dir), "wb") as f:
            pickle.dump(self.spectrum, f)

    def estimate_noise(self):
        data_len = len(self.corpus)
        data_1 = self.corpus[:data_len // 2]
        data_2 = self.corpus[data_len // 2 + 1:]
        matrix_1 = self.construct_matrix(data_1)
        matrix_2 = self.construct_matrix(data_2)
        diff = matrix_1 - matrix_2
        self.noise = np.std(diff) * 0.5

    def export_estimates(self):
        with open("{}/estimates.yml".format(self._param_dir), "w") as f:
            f.write("lambda: {}\n".format("sv.pkl"))
            f.write("sigma: {}\n".format(self.noise))
            f.write("alpha: {}\n".format(0.5)) #symmetric factorization


    def construct_matrix(self):
        raise NotImplementedError


