"""
算法：回文数字枚举
功能：xxx
题目：

===================================力扣===================================
2081. k 镜像数字的和（https://leetcode.cn/problems/sum-of-k-mirror-numbers/）枚举 10 进制回文数字并判断其 k 进制是否依然回文
866. 回文素数（https://leetcode.cn/problems/prime-palindrome/）枚举回文素数
564. 寻找最近的回文数（https://leetcode.cn/problems/find-the-closest-palindrome/）枚举字符的前半部分与后半部分
906. 超级回文数（https://leetcode.cn/problems/super-palindromes/）经典预处理所有的回文数其开方也是回文数
1088. 易混淆数 II（https://leetcode.cn/problems/confusing-number-ii/description/）模拟枚举所有的易混淆数

===================================洛谷===================================
P1609 最小回文数（https://www.luogu.com.cn/problem/P1609）枚举字符的前半部分与后半部分


参考：OI WiKi（xx）
"""
import bisect
from collections import defaultdict

from src.mathmatics.number_theory.template import NumberTheory
from src.strings.palindrome_num.template import PalindromeNum


class Solution:
    def __init__(self):
        return

    @staticmethod
    def lc_906(left: str, right: str) -> int:
        # 模板：经典预处理所有的回文数其开方也是回文数
        nums = PalindromeNum().get_palindrome_num_2(10)
        res = [num * num for num in nums if str(num * num)[::-1] == str(num * num)]
        left = int(left)
        right = int(right)
        return bisect.bisect_right(res, right) - bisect.bisect_left(res, left)

    @staticmethod
    def lc_1088(n: int) -> int:
        # 模板：预处理后进行二分查找
        ind = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}
        pre = [0, 1, 6, 8, 9]
        res = pre[:]
        for _ in range(2, 10):
            nex = []
            for num in pre:
                for d in ind:
                    # 经典枚举方式
                    nex.append(num * 10 + d)
            res.extend(nex)
            pre = nex[:]
        res = sorted(set(res))

        def check(x):
            # check函数
            if x <= 0:
                return False
            s = str(x)
            t = int("".join(str(ind[int(w)]) for w in s[::-1]))
            return t != x

        res = [num for num in res if check(num)]
        res.append(1000000000)  # 注意边界
        return bisect.bisect_right(res, n)

    @staticmethod
    def lc_2081(k: int, n: int) -> int:
        # 模板：枚举 10 进制回文数字并判断其 k 进制是否依然回文
        dct = defaultdict(list)
        # 放到预处理
        nums = PalindromeNum().get_palindrome_num_2(12)
        for k in range(2, 10):
            for num in nums:
                lst = NumberTheory().get_k_bin_of_n(num, k)
                if lst == lst[::-1]:
                    dct[k].append(num)
                    if len(dct[k]) >= 30:
                        break
        return sum(dct[k][:n])
