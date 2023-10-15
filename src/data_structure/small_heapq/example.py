

class TestGeneral(unittest.TestCase):

    def test_heapq_median(self):
        ceil = 1000
        num = random.randint(0, ceil)
        lst = SortedList([num])
        hm = HeapqMedian(num)
        for i in range(ceil):
            num = random.randint(0, ceil)
            lst.add(num)
            hm.add(num)
            if i % 2:
                assert lst[(i + 2) // 2] == hm.query()
        return


if __name__ == '__main__':
    unittest.main()
