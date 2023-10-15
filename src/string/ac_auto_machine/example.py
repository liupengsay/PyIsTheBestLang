

class TestGeneral(unittest.TestCase):

    def test_AhoCorasick(self):

        keywords = ["i","is", "ssippi"]
        auto = AhoCorasick(keywords)
        text = "misissippi"
        ans = auto.search_in(text)
        assert ans == {"i": [1, 3, 6, 9], "is": [1, 3], "ssippi": [4]}

        ac_tree = AhoCorasickAutomation()
        ac_tree.build_trie_tree(["i", "is", "ssippi"])
        ac_tree.build_ac_from_trie()
        res, dct = ac_tree.ac_search("misissippi")
        assert res == {"i": 4, "is": 2, "ssippi": 1}
        assert dct == ans
        return


if __name__ == '__main__':
    unittest.main()
