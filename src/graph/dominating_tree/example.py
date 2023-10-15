
class TestGeneral(unittest.TestCase):
    def test_dominate_tree(self):
        # 创建支配树对象
        dt = DominatingTree(5)

        # 添加图中的边
        dt.add_edge(0, 1)
        dt.add_edge(1, 2)
        dt.add_edge(2, 3)
        dt.add_edge(3, 4)

        # 构建支配树
        dt.build()

        # 获取每个点的支配点
        dominators = dt.get_dominators()

        # 输出每个点的支配点
        for i in range(5):
            print(f"点 {i} 的支配点为 {dominators[i]}")
        return


if __name__ == '__main__':
    unittest.main()
