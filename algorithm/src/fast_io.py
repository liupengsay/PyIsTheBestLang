import copy
import random
import heapq
import math
import sys
import bisect
import datetime
from functools import lru_cache
from collections import deque
from collections import Counter
from collections import defaultdict
from itertools import combinations
from itertools import permutations
import sys
from types import GeneratorType

import matplotlib.pyplot as plt
import networkx as nx
from functools import cmp_to_key
from functools import reduce
from operator import xor
from operator import mul
import networkx as nx
import matplotlib.pyplot as plt

from operator import add
from heapq import nlargest

inf = float("inf")


class FastIO:
    def __init__(self):
        self.inf = float("inf")
        return

    @staticmethod
    def _read():
        return sys.stdin.readline().strip()

    def read_int(self):
        return int(self._read())

    def read_float(self):
        return float(self._read())

    def read_ints(self):
        return map(int, self._read().split())

    def read_floats(self):
        return map(float, self._read().split())

    def read_ints_minus_one(self):
        return map(lambda x: int(x) - 1, self._read().split())

    def read_list_ints(self):
        return list(map(int, self._read().split()))

    def read_list_floats(self):
        return list(map(float, self._read().split()))

    def read_list_ints_minus_one(self):
        return list(map(lambda x: int(x) - 1, self._read().split()))

    def read_str(self):
        return self._read()

    def read_list_strs(self):
        return self._read().split()

    def read_list_str(self):
        return list(self._read())

    @staticmethod
    def st(x):
        return sys.stdout.write(str(x) + '\n')

    @staticmethod
    def lst(x):
        return sys.stdout.write(" ".join(str(w) for w in x) + '\n')

    @staticmethod
    def round_5(f):
        res = int(f)
        if f - res >= 0.5:
            res += 1
        return res

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def min(a, b):
        return a if a < b else b

    @staticmethod
    def bootstrap(f, queue=deque()):
        def wrappedfunc(*args, **kwargs):
            if queue:
                return f(*args, **kwargs)
            else:
                to = f(*args, **kwargs)
                while True:
                    if isinstance(to, GeneratorType):
                        queue.append(to)
                        to = next(to)
                    else:
                        queue.pop()
                        if not queue:
                            break
                        to = queue[-1].send(to)
                return to

        return wrappedfunc

    @staticmethod
    def plot_graph(n, edges):

        # 创建一个有向图
        g = nx.Graph()
        for i in range(n):
            # 添加节点
            g.add_node(i+1)
        for i in range(n):
            for j in edges[i]:
                g.add_edge(i+1, j+1)

        # for u, v in edges:
        #     # 添加边
        #     g.add_edge(u, v)

        # 绘制图形
        nx.draw_networkx(g)
        plt.show()
        return


