from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matrix.word2vec_matrix import Word2VecMatrix
from matrix.glove_matrix import GloVeMatrix
from matrix.ppmi_lsa_matrix import LSAMatrix

class SignalMatrixFactory():
    def __init__(self, corpus):
        self.corpus = corpus

    def produce(self, algo):
        if algo == "word2vec":
            return Word2VecMatrix(self.corpus)
        elif algo == "glove":
            return GloVeMatrix(self.corpus)
        elif algo == "lsa":
            return LSAMatrix(self.corpus)
        else:
            raise NotImplementedError


