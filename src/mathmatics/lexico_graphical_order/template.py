import math


class LexicoGraphicalOrder:
    def __init__(self):
        return

    @staticmethod
    def get_kth_num(n, k):
        # find the k-th smallest digit in the dictionary order within the range of 1 to n
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
        # Find the dictionary order of the number num within the range of 1 to n
        x = str(num)
        low = 1
        high = n
        while low < high - 1:
            # Using bisection for reverse engineering
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
        # The k-th smallest subset of set [1,...,n], with a total of 1<<n subsets
        # assert k <= (1 << n)
        ans = []
        if k == 1:
            # Empty subset output 0
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
        # Dictionary order of subsets lst of set [1,..., n]
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
        # Select the k-th comb of m elements from the set [1,...,n] to arrange the selection
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
        # The lexicographic order of selecting m elements in the set [1,...,n]

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
        # Select the k-th perm of n elements from the set [1,...,n] to arrange the perm selection
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
        # Dictionary order of perm permutation LST for n elements selected from set [1,...,n]
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
