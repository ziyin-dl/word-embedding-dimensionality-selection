from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import warnings

from matrix.signal_matrix import SignalMatrix

class Word2VecMatrix(SignalMatrix):

    def inject_params(self, kwargs):
        self._params = kwargs
        if "skip_window" not in self._params:
            self._params["skip_window"] = 5
        if "neg_samples" not in self._params:
            self._params["neg_samples"] = 1
        self.check_params()

    def check_params(self):
        if isinstance(self._params["skip_window"], int) and self._params["skip_window"] > 0:
            pass
        else:
            raise ValueError("skip_window must be a positive integer")
        if isinstance(self._params["neg_samples"], int) and self._params["neg_samples"] >= 0:
            self._params["neg_samples"] = max(self._params["neg_samples"], 1)
        else:
            raise ValueError("neg_samples must be a positive integer")

    def build_cooccurance_dict(self, data):
        skip_window = self._params["skip_window"]
        vocabulary_size = self.vocabulary_size
        cooccurance_count = collections.defaultdict(collections.Counter)
        for idx, center_word_id in enumerate(data):
            if center_word_id > vocabulary_size:
                vocabulary_size = center_word_id
            for i in range(max(idx - skip_window - 1, 0), min(idx + skip_window + 1, len(data))):
                cooccurance_count[center_word_id][data[i]] += 1
            cooccurance_count[center_word_id][center_word_id] -= 1
        return cooccurance_count, vocabulary_size


    def construct_matrix(self, data):
        cooccur, vocabulary_size = self.build_cooccurance_dict(data)
        k = self._params["neg_samples"]

        Nij = np.zeros([vocabulary_size, vocabulary_size])
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                Nij[i,j] += cooccur[i][j]
        Ni = np.sum(Nij, axis=1)
        tot = np.sum(Nij)
        with warnings.catch_warnings():
            """log(0) is going to throw warnings, but we will deal with it."""
            warnings.filterwarnings("ignore")
            Pij = Nij / tot 
            Pi = Ni / np.sum(Ni)
            # c.f.Neural Word Embedding as Implicit Matrix Factorization, Levy & Goldberg, 2014
            PMI = np.log(Pij) - np.log(np.outer(Pi, Pi)) - np.log(k)
            PMI[np.isinf(PMI)] = 0
            PMI[np.isnan(PMI)] = 0
        return PMI

