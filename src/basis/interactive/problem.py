"""
算法：二分查找、交互题
功能：
参考：
题目：

===================================力扣===================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

===================================洛谷===================================
xx（xxx）xxxxxxxxxxxxxxxxxxxx

================================CodeForces================================
F. Interacdive Problem（https://codeforces.com/contest/1624/problem/F）二分查找交互
D. Tournament Countdown（https://codeforces.com/contest/1713/problem/D）
D. Guess The String（https://codeforces.com/contest/1697/problem/D）经典二分查找交互，严格非红蓝二分写法


"""
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1697d(ac=FastIO()):
        n = ac.read_int()
        ans = [""] * n
        dct = dict()

        ac.lst(["?", 1, 1])
        sys.stdout.flush()
        w = ac.read_str()
        ans[0] = w
        dct[w] = 0

        pre = 1
        for i in range(1, n):
            ac.lst(["?", 2, 1, i + 1])
            sys.stdout.flush()
            cur = ac.read_int()
            if cur > pre:

                ac.lst(["?", 1, i + 1])
                sys.stdout.flush()
                w = ac.read_str()
                ans[i] = w
                dct[w] = i

                pre = cur
            else:
                lst = sorted(dct.values())
                m = len(lst)
                low, high = 0, m - 1
                while low < high:
                    mid = low + (high - low + 1) // 2
                    target = len(set(ans[lst[mid]:i]))

                    ac.lst(["?", 2, lst[mid] + 1, i + 1])
                    sys.stdout.flush()
                    cur = ac.read_int()
                    if cur == target:
                        low = mid
                    else:
                        high = mid - 1
                ans[i] = ans[lst[high]]
                dct[ans[i]] = i
        ac.lst(["!", "".join(ans)])
        sys.stdout.flush()
        return
