from functools import reduce
from operator import xor


class Nim:
    def __init__(self, lst):
        self.lst = lst
        return

    def gen_result(self):
        return reduce(xor, self.lst) != 0
