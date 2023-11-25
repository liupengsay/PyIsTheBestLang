import math
from functools import reduce


class PeiShuTheorem:
    def __init__(self):
        return

    @staticmethod
    def get_lst_gcd(lst):
        return reduce(math.gcd, lst)
