class ManacherPlindrome:
    def __init__(self):
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def manacher(s):
        # 马拉车算法
        n = len(s)
        arm = [0] * n
        left, right = 0, -1
        for i in range(0, n):
            a, b = arm[left + right - i], right - i + 1
            a = a if a < b else b
            k = 1 if i > right else a

            # 持续增加回文串的长度
            while 0 <= i - k and i + k < n and s[i - k] == s[i + k]:
                k += 1
            arm[i] = k

            # 更新右侧最远的回文串边界
            k -= 1
            if i + k > right:
                left = i - k
                right = i + k
        # 返回每个位置往右的臂长其中 s[i-arm[i]+1: i+arm[i]] 为回文子串范围
        # 即以i为中心的最长回文子串长度
        return arm

    def palindrome(self, s: str) -> (list, list):
        # 获取区间的回文串信息
        n = len(s)
        # 保证所有的回文串为奇数长度，且中心为 # 的为原偶数回文子串，中心为 字母 的为原奇数回文子串
        t = "#" + "#".join(list(s)) + "#"
        dp = self.manacher(t)
        m = len(t)

        # 以当前索引作为边界开头的回文子串结束位置索引
        start = [[] for _ in range(n)]
        # 以当前索引作为边界结尾的回文子串起始位置索引
        end = [[] for _ in range(n)]

        for j in range(m):
            left = j - dp[j] + 1
            right = j + dp[j] - 1
            while left <= right:
                if t[left] != "#":
                    start[left // 2].append(right // 2)
                    end[right // 2].append(left // 2)
                left += 1
                right -= 1
        return start, end

    def palindrome_longest(self, s: str) -> (list, list):
        # 获取区间的回文串信息
        n = len(s)
        # 保证所有的回文串为奇数长度，且中心为 # 的为原偶数回文子串，中心为 字母 的为原奇数回文子串
        t = "#" + "#".join(list(s)) + "#"
        dp = self.manacher(t)
        m = len(t)

        # 由此还可以获得以某个位置开头或者结尾的最长回文子串的长度
        post = [1] * n
        pre = [1] * n

        for j in range(m):
            left = j - dp[j] + 1
            right = j + dp[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    post[x] = self.max(post[x], y - x + 1)
                    pre[y] = self.max(pre[y], y - x + 1)
                    break
                left += 1
                right -= 1
        # 由此还可以获得以某个位置开头或者结尾的最长回文子串
        for i in range(1, n):
            if i - pre[i - 1] - 1 >= 0 and s[i] == s[i - pre[i - 1] - 1]:
                pre[i] = self.max(pre[i], pre[i - 1] + 2)

        for i in range(n - 2, -1, -1):
            pre[i] = self.max(pre[i], pre[i + 1] - 2)

        for i in range(n - 2, -1, -1):
            if i + post[i + 1] + 1 < n and s[i] == s[i + post[i + 1] + 1]:
                post[i] = self.max(post[i], post[i + 1] + 2)
        for i in range(1, n):
            post[i] = self.max(post[i], post[i - 1] - 2)

        return post, pre

    def palindrome_longest_length(self, s: str) -> (list, list):
        # 计算字符串的最长回文子串长度
        t = "#" + "#".join(list(s)) + "#"
        dp = self.manacher(t)
        m = len(t)
        ans = 0
        for j in range(m):
            left = j - dp[j] + 1
            right = j + dp[j] - 1
            cur = (right - left + 1) // 2
            ans = ans if ans > cur else cur
        return ans
