class PalindromeNum:
    def __init__(self):
        return

    @staticmethod
    def get_palindrome_num_1(n):
        """template of get all positive palindrome number not greater than n"""
        dp = [[""], [str(i) for i in range(10)]]
        for k in range(2, n + 1):
            # like dp to add palindrome character
            if k % 2 == 1:
                m = k // 2
                lst = []
                for st in dp[-1]:
                    for i in range(10):
                        lst.append(st[:m] + str(i) + st[m:])
                dp.append(lst)
            else:
                lst = []
                for st in dp[-2]:
                    for i in range(10):
                        lst.append(str(i) + st + str(i))
                dp.append(lst)

        nums = []
        for lst in dp:
            for num in lst:
                if num and num[0] != "0":
                    nums.append(int(num))
        nums.sort()
        return nums

    @staticmethod
    def get_palindrome_num_2(n):
        assert n >= 1
        """template of get all positive palindrome number whose length not greater than n"""
        nums = list(range(1, 10))
        x = 1
        while len(str(x)) * 2 <= n:
            num = str(x) + str(x)[::-1]
            nums.append(int(num))
            if len(str(x)) * 2 + 1 <= n:
                for d in range(10):
                    nums.append(int(str(x) + str(d) + str(x)[::-1]))
            x += 1
        nums.sort()
        return nums

    @staticmethod
    def get_recent_palindrome_num(n: str) -> list:
        """template of recentest palindrome num of n"""
        m = len(n)
        candidates = [10 ** (m - 1) - 1, 10 ** m + 1]
        prefix = int(n[:(m + 1) // 2])
        for x in range(prefix - 1, prefix + 2):
            y = x if m % 2 == 0 else x // 10
            while y:
                x = x * 10 + y % 10
                y //= 10
            candidates.append(x)
        return candidates
