import math


class LexicoGraphicalOrder:
    def __init__(self):
        return

    @staticmethod
    def get_kth_num(n, k):
        # 模板：求 1 到 n 范围内字典序第 k 小的数字
        def check():
            c = 0
            first = last = num
            while first <= n:
                c += min(last, n) - first + 1
                last = last * 10 + 9
                first *= 10
            return c

        # assert k <= n
        num = 1
        k -= 1
        while k:
            cnt = check()
            if k >= cnt:
                num += 1
                k -= cnt
            else:
                num *= 10
                k -= 1
        return num

    def get_num_kth(self, n, num):
        # 模板：求 1 到 n 范围内数字 num 的字典序
        x = str(num)
        low = 1
        high = n
        while low < high - 1:
            # 使用二分进行逆向工程
            mid = low + (high - low) // 2
            st = str(self.get_kth_num(n, mid))
            if st < x:
                low = mid
            elif st > x:
                high = mid
            else:
                return mid
        return low if str(self.get_kth_num(n, low)) == x else high

    @staticmethod
    def get_kth_subset(n, k):

        # 集合 [1,..,n] 的第 k 小的子集，总共有 1<<n 个子集
        # assert k <= (1 << n)
        ans = []
        if k == 1:
            # 空子集输出 0
            ans.append(0)
        k -= 1
        for i in range(1, n + 1):
            if k == 0:
                break
            if k <= pow(2, n - i):
                ans.append(i)
                k -= 1
            else:
                k -= pow(2, n - i)
        return ans

    def get_subset_kth(self, n, lst):

        # 集合 [1,..,n] 的子集 lst 的字典序
        low = 1
        high = n
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset(n, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset(n, low) == lst else high

    @staticmethod
    def get_kth_subset_comb(n, m, k):
        # 集合 [1,..,n] 中选取 m 个元素的第 k 个 comb 选取排列
        # assert k <= math.comb(n, m)

        nums = list(range(1, n + 1))
        ans = []
        while k and nums and len(ans) < m:
            length = len(nums)
            c = math.comb(length - 1, m - len(ans) - 1)
            if c >= k:
                ans.append(nums.pop(0))
            else:
                k -= c
                nums.pop(0)
        return ans

    def get_subset_comb_kth(self, n, m, lst):
        # 集合 [1,..,n] 中选取 m 个元素的排列 lst 的字典序

        low = 1
        high = math.comb(n, m)
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset_comb(n, m, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset_comb(n, m, low) == lst else high

    @staticmethod
    def get_kth_subset_perm(n, k):
        # 集合 [1,..,n] 中选取 n 个元素的第 k 个 perm 选取排列
        s = math.factorial(n)
        assert 1 <= k <= s
        nums = list(range(1, n + 1))
        ans = []
        while k and nums:
            single = s//len(nums)
            i = (k - 1) // single
            ans.append(nums.pop(i))
            k -= i * single
            s = single
        return ans

    def get_subset_perm_kth(self, n, lst):
        # 集合 [1,..,n] 中选取 n 个元素的 perm 全排列 lst 的字典序

        low = 1
        high = math.factorial(n)
        while low < high - 1:
            mid = low + (high - low) // 2
            st = self.get_kth_subset_perm(n, mid)
            if st < lst:
                low = mid
            elif st > lst:
                high = mid
            else:
                return mid
        return low if self.get_kth_subset_perm(n, low) == lst else high
