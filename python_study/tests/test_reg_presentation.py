import unittest
import re


class TestGeneral(unittest.TestCase):
    def test_sub(self):
        error = 'This iss my first case!'
        correct = 'This iss my first case!'
        pattern = r'is[2,]'
        self.assertEqual(re.sub(pattern, 'is', error),
                         correct)
        return


if __name__ == '__main__':
    unittest.main()
