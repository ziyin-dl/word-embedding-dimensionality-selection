from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import subprocess as sp
from multiprocessing import Pool

def _lower(s):
    return s.lower()

class SimpleTokenizer():
    def __init__(self):
        pass

    def tokenize(self, data):    
        """data: str"""
        splitted = data.split(' ')
        pool = Pool()
        tokenized = pool.map(_lower, splitted)
        return tokenized

    #TODO: add min_count, together with n_words to determine if UNK is needed
    def frequency_count(self, tokenized_data, n_words):
        count = [['UNK', -1]]
        counter = collections.Counter(tokenized_data)
        # if more tokens than needed, map the rest to UNK
        if len(counter) > n_words:
            count.extend(collections.Counter(tokenized_data).most_common(n_words - 1))
        else:
            count = collections.Counter(tokenized_data).most_common(n_words)
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reversed_dictionary

    def index(self, tokenized_data, dictionary):
        data = list()
        unk_count = 0
        
        def _index(word):
            if word in dictionary:
                index = dictionary[word]
            else:
                index = dictionary['UNK']
            return index

        data = [_index(word) for word in tokenized_data]
        return data
    
    def do_index_data(self, data, n_words=10000):
        """transform data: str into a tokens: list. tokens are mapped to {0, 1, ..., n_words - 1}"""
        self.tokenized = self.tokenize(data)
        self.dictionary, self.reversed_dictionary = self.frequency_count(self.tokenized, n_words)
        self.indexed = self.index(self.tokenized, self.dictionary)
        return self.indexed
