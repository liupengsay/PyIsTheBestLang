class ManacherPlindrome:
    def __init__(self):
        return

    @staticmethod
    def max(a, b):
        return a if a > b else b

    @staticmethod
    def manacher(s):
        """template of get the palindrome radius for every i-th character as center"""
        n = len(s)
        arm = [0] * n
        left, right = 0, -1
        for i in range(0, n):
            a, b = arm[left + right - i], right - i + 1
            a = a if a < b else b
            k = 1 if i > right else a
            while 0 <= i - k and i + k < n and s[i - k] == s[i + k]:
                k += 1
            arm[i] = k
            k -= 1
            if i + k > right:
                left = i - k
                right = i + k
        # s[i-arm[i]+1: i+arm[i]] is palindrome substring for every i
        return arm

    def palindrome(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        dp = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        start = [[] for _ in range(n)]
        # the starting position index of the palindrome substring ending with the current index as the boundary
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
        """template of get the length of the longest palindrome substring that starts or ends at a certain position"""
        n = len(s)
        t = "#" + "#".join(list(s)) + "#"
        dp = self.manacher(t)
        m = len(t)
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
        """template of get the longest palindrome substring of s"""
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
