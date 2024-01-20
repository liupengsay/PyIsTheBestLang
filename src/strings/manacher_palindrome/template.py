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
            k = 0 if i > right else a
            while 0 <= i - k and i + k < n and s[i - k] == s[i + k]:
                k += 1
            arm[i] = k
            k -= 1
            if i + k > right:
                left = i - k
                right = i + k
        # s[i-arm[i]+1: i+arm[i]] is palindrome substring for every i
        return arm

    def palindrome_start_end(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        start = [[] for _ in range(n)]
        # the starting position index of the palindrome substring ending with the current index as the boundary
        end = [[] for _ in range(n)]
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    start[left // 2].append(right // 2)
                    end[right // 2].append(left // 2)
                left += 1
                right -= 1
        return start, end

    def palindrome_post_pre(self, s: str) -> (list, list):
        """template of get the length of the longest palindrome substring that starts or ends at a certain position"""
        n = len(s)
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)
        post = [1] * n
        pre = [1] * n
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
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
        arm = self.manacher(t)
        m = len(t)
        ans = 0
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            cur = (right - left + 1) // 2
            ans = ans if ans > cur else cur
        return ans

    def palindrome_just_start(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        start = []
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    if left // 2 == 0:
                        start.append(right // 2)
                    break
                left += 1
                right -= 1
        return start  # prefix palindrome

    def palindrome_just_end(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        end = []
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    if right // 2 == n-1:
                        end.append(left // 2)
                    break
                left += 1
                right -= 1
        return end  # suffix palindrome

    def palindrome_count_start_end(self, s: str) -> (list, list):
        """template of get the number of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        start = [0] * n
        end = [0] * n
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    if (y - x + 1) % 2:
                        mid = x + (y - x + 1) // 2
                        start[x] += 1
                        if mid + 1 < n:
                            start[mid + 1] -= 1
                        end[mid] += 1
                        if y + 1 < n:
                            end[y + 1] -= 1
                    else:
                        mid = x + (y - x + 1) // 2 - 1
                        start[x] += 1
                        start[mid + 1] -= 1
                        end[mid + 1] += 1
                        if y + 1 < n:
                            end[y + 1] -= 1
                    break
                left += 1
                right -= 1
        for i in range(1, n):
            start[i] += start[i - 1]
            end[i] += end[i - 1]
        return start, end

    def palindrome_count_start_end_odd(self, s: str) -> (list, list):
        """template of get the number of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        start = [0] * n
        end = [0] * n
        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    if (y - x + 1) % 2:
                        mid = x + (y - x + 1) // 2
                        start[x] += 1
                        if mid + 1 < n:
                            start[mid + 1] -= 1
                        end[mid] += 1
                        if y + 1 < n:
                            end[y + 1] -= 1
                    break
                left += 1
                right -= 1
        for i in range(1, n):
            start[i] += start[i - 1]
            end[i] += end[i - 1]
        return start, end


    def palindrome_length_count(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        n = len(s)
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        odd = [0] * (n + 2)
        even = [0] * (n + 2)

        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    if (y - x + 1) % 2:
                        low, high = 1, (y - x + 2) // 2
                        odd[low] += 1
                        if high + 1 <= n:
                            odd[high + 1] -= 1
                    else:
                        low, high = 1, (y - x + 1) // 2
                        even[low] += 1
                        if high + 1 <= n:
                            even[high + 1] -= 1
                    break
                left += 1
                right -= 1
        cnt = [0] * (n + 1)
        for i in range(1, n + 1):
            odd[i] += odd[i - 1]
            even[i] += even[i - 1]
            if 2 * i - 1 <= n:
                cnt[2 * i - 1] += odd[i]
            if 2 * i <= n:
                cnt[2 * i] += even[i]
        return cnt

    def palindrome_count(self, s: str) -> (list, list):
        """template of get the endpoint of palindrome substring for every i-th character as start or end pos"""
        # trick to promise every palindrome substring has odd length
        # with # centered as the original even palindrome substring
        # letter centered as the original odd palindrome substring
        t = "#" + "#".join(list(s)) + "#"
        arm = self.manacher(t)
        m = len(t)

        # end position index of palindrome substring starting with the current index as the boundary
        ans = 0

        for j in range(m):
            left = j - arm[j] + 1
            right = j + arm[j] - 1
            while left <= right:
                if t[left] != "#":
                    x, y = left // 2, right // 2
                    if (y - x + 1) % 2:
                        ans += (y-x+2)//2
                    else:
                        ans += (y-x+1)//2
                    break
                left += 1
                right -= 1
        return ans
