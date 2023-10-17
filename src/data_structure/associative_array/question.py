from utils.fast_io import FastIO


"""
Algorithm：use xor of random seed as key of mapping
Ability：speed up and avoid hash crush
Reference：https://judge.yosupo.jp/problem/associative_array

================================Library Checker================================
Associative Array（https://judge.yosupo.jp/problem/associative_array）use xor of random seed as key of mapping


"""


class Solution:
    def __int__(self):
        return

    @staticmethod
    def libc_aa(ac=FastIO()):
        """template question of associative array"""
        pre = dict()
        for _ in range(ac.read_int()):
            lst = ac.read_list_ints()
            if lst[0] == 0:
                k, v = lst[1:]
                pre[k ^ ac.random_seed] = v
            else:
                ac.st(pre.get(lst[1] ^ ac.random_seed, 0))
        return
