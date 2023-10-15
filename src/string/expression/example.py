

class TestGeneral(unittest.TestCase):

    def test_tree_expression(self):
        te = TreeExpression()
        lst = ["2*3^4^2+(5/2-2)", "-2+3", "2*(-5/2+3*2)-33", "((-2+3)*3+5-7/2)^2", "2*(-3)", "0-(-3)", "-(-3)+2"]
        for s in lst:
            assert int(te.main_1175(s)[-1][0]) == eval(s.replace("^", "**").replace("/", "//"))
        return

    def test_english_number(self):

        en = EnglishNumber()
        num = 5208
        assert en.number_to_english(num) == "five thousand two hundred and eight"



if __name__ == '__main__':
    unittest.main()
