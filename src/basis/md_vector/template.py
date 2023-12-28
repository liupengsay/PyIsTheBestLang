from functools import reduce
from operator import mul


class MdVector:
    def __init__(self, dimension, initial):
        self.dimension = dimension
        self.dp = [initial] * reduce(mul, dimension)
        self.m = len(dimension)
        self.pos = []
        for i in range(self.m):
            self.pos.append(reduce(mul, dimension[i + 1:] + [1]))
        return

    def get(self, lst):
        return sum(x * y for x, y in zip(lst, self.pos))
