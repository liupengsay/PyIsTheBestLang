import math
import unittest
from functools import reduce
from typing import List

from src.fast_io import FastIO



class PeiShuTheorem:
    def __init__(self):
        return

    @staticmethod
    def get_lst_gcd(lst):
        return reduce(math.gcd, lst)

