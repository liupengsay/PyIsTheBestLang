import math
import random
import unittest
from functools import reduce
from math import gcd
from operator import add
from itertools import accumulate
from typing import List
from operator import mul, add, xor, and_, or_

"""
多进程处理数据
"""


import time
from multiprocessing import Pool


class MultiProcessingPool:
    def __init__(self):
        return

    @staticmethod
    def run(num):
        # num: 函数参数是数据列表的一个元素
        time.sleep(0.01)
        return num*num


class TestGeneral(unittest.TestCase):

    def test_multiprocessing_pool(self):
        print("\n")
        params = list(range(1000))
        mp = MultiProcessingPool()
        print("in order:")  # 顺序执行(也就是串行执行，单进程)
        s1 = time.time()
        for num in params:
            mp.run(num)
        t1 = time.time()
        print("顺序执行时间：", int(t1 - s1))

        print("in pool:")  # 创建多个进程，并行执行
        pool = Pool(16)  # 创建拥有16个进程数量的进程池
        # run：处理testFL列表中数据的函数 params:要处理的数据列表
        pool.map(mp.run, params)
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出
        t2 = time.time()
        print("并行执行时间：", int(t2 - t1))
        return


if __name__ == '__main__':
    unittest.main()
