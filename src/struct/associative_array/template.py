import random
from collections import Counter


class HashWithRandomSeedEscapeExplode:
    def __int__(self):
        return

    @staticmethod
    def get_cnt(nums):
        """template of associative array"""
        seed = random.randint(0, 10 ** 9 + 7)
        return Counter([num ^ seed for num in nums])
