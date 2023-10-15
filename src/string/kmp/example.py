





class TestGeneral(unittest.TestCase):

    def test_prefix_function(self):
        kmp = KMP()
        assert kmp.prefix_function("abcabcd") == [0, 0, 0, 1, 2, 3, 0]
        assert kmp.prefix_function("aabaaab") == [0, 1, 0, 1, 2, 2, 3]
        return

    def test_z_function(self):
        kmp = KMP()
        assert kmp.z_function("abacaba") == [0, 0, 1, 0, 3, 0, 1]
        assert kmp.z_function("aaabaab") == [0, 2, 1, 0, 2, 1, 0]
        assert kmp.z_function("aaaaa") == [0, 4, 3, 2, 1]
        return

    def test_find(self):
        kmp = KMP()
        assert kmp.find("abababc", "aba") == [0, 2]
        assert kmp.find("aaaa", "a") == [0, 1, 2, 3]
        assert kmp.find("aaaa", "aaaa") == [0]
        return


if __name__ == '__main__':
    unittest.main()
