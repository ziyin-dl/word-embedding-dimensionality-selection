from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zipfile
try:
    import cPickle as pickle
except ImportError:
    import pickle
import tensorflow as tf

class ZipFileReader():
    def __init__(self):
        pass

    def read_data(self, filename):
        """Extract the first file enclosed in a zip file as a string."""
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0]))
        return data

class RawTextReader():
    def __init__(self):
        pass

    def read_data(self, filename):
        with open(filename, 'r') as f:
            data = f.read()
        return data

class ReaderFactory():
    def __init__(self):
        pass

    @classmethod
    def produce(self, filetype):
        if filetype == "zip":
            return ZipFileReader()
        else:
            return RawTextReader()

