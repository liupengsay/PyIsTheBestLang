"""
Algorithm：palindrome_number|brute_force
Description：

====================================LeetCode====================================
2081（https://leetcode.com/problems/sum-of-k-mirror-numbers/）brute_force|10-base|palindrome_number
866（https://leetcode.com/problems/prime-palindrome/）brute_force|palindrome_prime
564（https://leetcode.com/problems/find-the-closest-palindrome/）brute_force
906（https://leetcode.com/problems/super-palindromes/）preprocess|brute_force
1088（https://leetcode.com/problems/confusing-number-ii/description/）implemention|brute_force

=====================================LuoGu======================================
P1609（https://www.luogu.com.cn/problem/P1609）brute_force


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
        """
        url: https://leetcode.com/problems/super-palindromes/
        tag: preprocess|brute_force
        """
        # preprocess所有的回文数其开方也是回文数
        nums = PalindromeNum().get_palindrome_num_2(10)
        res = [num * num for num in nums if str(num * num)[::-1] == str(num * num)]
        left = int(left)
        right = int(right)
        return bisect.bisect_right(res, right) - bisect.bisect_left(res, left)

    @staticmethod
    def lc_1088(n: int) -> int:
        """
        url: https://leetcode.com/problems/confusing-number-ii/description/
        tag: implemention|brute_force
        """
        # preprocess后binary_search
        ind = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}
        pre = [0, 1, 6, 8, 9]
        res = pre[:]
        for _ in range(2, 10):
            nex = []
            for num in pre:
                for d in ind:
                    # brute_force方式
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
        """
        url: https://leetcode.com/problems/sum-of-k-mirror-numbers/
        tag: brute_force|10-base|palindrome_number
        """
        # brute_force 10 进制palindrome_number并判断其 k 进制是否依然回文
        dct = defaultdict(list)
        # 放到preprocess
        nums = PalindromeNum().get_palindrome_num_2(12)
        for k in range(2, 10):
            for num in nums:
                lst = NumberTheory().get_k_bin_of_n(num, k)
                if lst == lst[::-1]:
                    dct[k].append(num)
                    if len(dct[k]) >= 30:
                        break
        return sum(dct[k][:n])