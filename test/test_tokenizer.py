import unittest
import utils.tokenizer as tokenizer
import utils.reader as reader

from mock import patch, mock_open

class TestRawTextReader(unittest.TestCase):
    def setUp(self, string="a b C d Ef G"):
        self._string = string
        self._reader = reader.RawTextReader()

    def test_read(self):
        m = mock_open()
        with patch('utils.reader.open',  mock_open(read_data=self._string), create=True):
            data = self._reader.read_data('fake_test.txt')
        self.assertEqual(self._string, data)
        #m.assert_called_once_with('fake_test.txt')
    
class TestTokenizer(unittest.TestCase):
    def setUp(self, string="a b C d Ef G"):
        self._expected_tokens = ["a", "b", "c", "d", "ef", "g"]
        self._expected_counts = {"a": 1, "b": 1, "c": 1, "d": 1, "ef": 1, "g": 1}
        self._string = string
        self._tokenizer = tokenizer.SimpleTokenizer()

    def test_tokenizer(self):
        ret = self._tokenizer.tokenize(self._string)
        self.assertEqual(self._expected_tokens, ret)

    def test_index(self):
        tokens = self._tokenizer.tokenize(self._string)
        dic, rev_dic = self._tokenizer.frequency_count(tokens, 10000)
        ret = self._tokenizer.index(tokens, dic)
        expected_indices = [dic[w] for w in tokens]
        self.assertEqual(expected_indices, ret)

if __name__ == "__main__":
    unittest.main()
