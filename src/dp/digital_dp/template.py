from functools import lru_cache


class DigitalDP:
    def __init__(self):
        return

    @staticmethod
    def count_bin(n):
        # calculate the number of occurrences of positive integer binary bit 1 from 1 to n

        @lru_cache(None)
        def dfs(i, is_limit, is_num, cnt):
            if i == m:
                if is_num:
                    return cnt
                return 0
            res = 0
            if not is_num:
                res += dfs(i + 1, False, False, cnt)
            low = 0 if is_num else 1
            high = int(st[i]) if is_limit else 1
            for x in range(low, high + 1):
                res += dfs(i + 1, is_limit and high == x, True, cnt + int(i == w) * x)
            return res

        st = bin(n)[2:]
        m = len(st)
        ans = []  # From binary high to binary low
        for w in range(m):
            cur = dfs(0, True, False, 0)
            ans.append(cur)
            dfs.cache_clear()
        return ans

    @staticmethod
    def count_digit(num, d):
        # Calculate the number of occurrences of digit d within 1 to num

        @lru_cache(None)
        def dfs(i, cnt, is_limit, is_num):
            if i == n:
                if is_num:
                    return cnt
                return 0
            res = 0
            if not is_num:
                res += dfs(i + 1, 0, False, False)

            floor = 0 if is_num else 1
            ceil = int(s[i]) if is_limit else 9
            for x in range(floor, ceil + 1):
                res += dfs(i + 1, cnt + int(x == d), is_limit and ceil == x, True)
            return res

        s = str(num)
        n = len(s)
        return dfs(0, 0, True, False)

    @staticmethod
    def count_digit_sum(num):
        # Calculate the number of occurrences of digit d within 1 to num

        @lru_cache(None)
        def dfs(i, cnt, is_limit, is_num):
            if i == n:
                if is_num:
                    return cnt
                return 0
            res = 0
            if not is_num:
                res += dfs(i + 1, 0, False, False)

            floor = 0 if is_num else 1
            ceil = int(s[i]) if is_limit else 9
            for x in range(floor, ceil + 1):
                res += dfs(i + 1, cnt + x, is_limit and ceil == x, True)
            return res

        s = str(num)
        n = len(s)
        return dfs(0, 0, True, False)

    @staticmethod
    def count_digit_bfs(num, d):
        # Calculate the number of occurrences of digit d within 1 to num
        s = str(num)
        n = len(s)
        dp = dict()
        sub = [[]]
        ind = 1
        stack = [(0, 0, 1, False, 0)]
        while stack:
            i, cnt, is_limit, is_num, x = stack.pop()
            if i >= 0:
                if i == n:
                    dp[x] = cnt if is_num else 0
                    continue
                stack.append((~i, cnt, is_limit, is_num, x))
                if not is_num:
                    stack.append((i + 1, 0, False, False, ind))
                    sub.append([])
                    sub[x].append(ind)
                    ind += 1

                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for xx in range(floor, ceil + 1):
                    stack.append((i + 1, cnt + int(xx == d), is_limit and ceil == xx, True, ind))
                    sub.append([])
                    sub[x].append(ind)
                    ind += 1
            else:
                dp[x] = sum(dp[y] for y in sub[x])
        return dp[0]

    @staticmethod
    def count_digit_iteration(num, d):
        # Calculate the number of occurrences of digit d within 1 to num by iteration
        assert num >= 1
        s = str(num)
        n = len(s)

        dp = [[[[0] * 2 for _ in range(2)] for _ in range(n + 2)] for _ in range(n + 1)]
        for i in range(n, -1, -1):
            for cnt in range(n, -1, -1):
                for is_limit in range(1, -1, -1):
                    for is_num in range(1, -1, -1):
                        if i == n:
                            dp[i][cnt][is_limit][is_num] = cnt if is_num else 0
                            continue
                        res = 0
                        if not is_num:
                            res += dp[i + 1][0][0][0]
                        floor = 0 if is_num else 1
                        ceil = int(s[i]) if is_limit else 9
                        for x in range(floor, ceil + 1):
                            res += dp[i + 1][cnt + int(x == d)][int(is_limit and x == ceil)][1]
                        dp[i][cnt][is_limit][is_num] = res
        return dp[0][0][1][0]

    @staticmethod
    def count_digit_iteration_md(num, d):
        # Calculate the number of occurrences of digit d within 1 to num by iteration
        assert num >= 1
        s = str(num)
        n = len(s)

        def pos_to_ind(i1, i2, i3, i4):
            return i1 * (n + 2) * 2 * 2 + i2 * 2 * 2 + i3 * 2 + i4

        dp = [0] * (n + 1) * (n + 2) * 2 * 2
        for i in range(n, -1, -1):
            for cnt in range(n, -1, -1):
                for is_limit in range(1, -1, -1):
                    for is_num in range(1, -1, -1):
                        if i == n:
                            dp[pos_to_ind(i, cnt, is_limit, is_num)] = cnt if is_num else 0
                            continue
                        res = 0
                        if not is_num:
                            res += dp[pos_to_ind(i + 1, 0, 0, 0)]
                        floor = 0 if is_num else 1
                        ceil = int(s[i]) if is_limit else 9
                        for x in range(floor, ceil + 1):
                            res += dp[pos_to_ind(i + 1, cnt + int(x == d), int(is_limit and x == ceil), 1)]
                        dp[pos_to_ind(i, cnt, is_limit, is_num)] = res
        return dp[pos_to_ind(0, 0, 1, 0)]

    @staticmethod
    def count_num_base(num, d):
        # Use decimal to calculate the number of digits from 1 to num without the digit d
        assert 1 <= d <= 9  # If 0 is not included, use digital DP for calculation
        s = str(num)
        i = s.find(str(d))
        if i != -1:
            if d:
                s = s[:i] + str(d - 1) + (len(s) - i - 1) * "9"
            else:
                s = s[:i - 1] + str(int(s[i - 1]) - 1) + (len(s) - i - 1) * "9"
            num = int(s)

        lst = []
        while num:
            lst.append(num % 10)
            if d and lst[-1] >= d:
                lst[-1] -= 1
            elif not d and lst[-1] == 0:
                num *= 10
                num -= 1
                lst.append(num % 10)
            num //= 10
        lst.reverse()

        ans = 0
        for x in lst:
            ans *= 9
            ans += x
        return ans

    @staticmethod
    def count_num_dp(num, d):

        # Use decimal to calculate the number of digits from 1 to num without the digit d
        assert 0 <= d <= 9

        @lru_cache(None)
        def dfs(i: int, is_limit: bool, is_num: bool) -> int:
            if i == m:
                return int(is_num)

            res = 0
            if not is_num:
                res = dfs(i + 1, False, False)
            up = int(s[i]) if is_limit else 9
            for x in range(0 if is_num else 1, up + 1):
                if x != d:
                    res += dfs(i + 1, is_limit and x == up, True)
            return res

        s = str(num)
        m = len(s)
        return dfs(0, True, False)

    @staticmethod
    def get_kth_without_d(k, d):
        # Use decimal to calculate the k-th digit without digit d 0<=d<=9
        assert 0 <= d <= 9
        lst = []
        st = list(range(10))
        st.remove(d)
        while k:
            if d:
                lst.append(k % 9)
                k //= 9
            else:
                lst.append((k - 1) % 9)
                k = (k - 1) // 9
        lst.reverse()
        # It can also be solved using binary search and digit DP
        ans = [str(st[i]) for i in lst]
        return int("".join(ans))
