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
