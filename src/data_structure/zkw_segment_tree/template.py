class PointSetRangeSum:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [initial] * 2 * self.n
        return

    def push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

    def build(self, nums):
        for i in range(self.n):
            self.cover[i + self.n] = nums[i]
        for i in range(self.n - 1, 0, -1):
            self.push_up(i)
        return

    def get(self):
        return self.cover[self.n:]

    def point_set(self, ind, val):
        ind += self.n
        self.cover[ind] = val
        while ind > 1:
            ind //= 2
            self.push_up(ind)
        return

    def range_sum(self, left, right):
        ans = 0
        left += self.n
        right += self.n + 1
        while left < right:
            if left & 1:
                ans += self.cover[left]
                left += 1
            if right & 1:
                right -= 1
                ans += self.cover[right]
            left >>= 1
            right >>= 1
        return ans


class PointSetRangeSumStack:

    def __init__(self, n, initial=0):
        self.n = n
        self.initial = initial
        self.cover = [initial] * (2 * n)
        self.m = 1
        while self.m < self.n:
            self.m *= 2

        return

    def push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

    def build(self, nums):
        stack = [1]
        while stack:
            i = stack.pop()
            if i >= self.n * 2:
                continue
            if i >= self.n:
                self.cover[i] = nums[i - self.n]
            else:
                stack.append(i << 1)
                stack.append((i << 1) | 1)
        for i in range(self.n - 1, 0, -1):
            self.push_up(i)
        return

    def get(self):
        nums = [0] * self.n
        stack = [1]
        while stack:
            i = stack.pop()
            if i >= self.n * 2:
                continue
            if i >= self.n:
                nums[i - self.n] = self.cover[i]
            else:
                stack.append(i << 1)
                stack.append((i << 1) | 1)
        return nums

    def point_set(self, ind, val):
        ind += self.n
        self.cover[ind] = val
        while ind > 1:
            ind //= 2
            self.push_up(ind)
        return

    def range_sum(self, left, right):

        ans = 0
        left += self.n
        right += self.n
        stack = [(self.m, self.m * 2 - 1, 1)]
        while stack:
            s, t, i = stack.pop()
            if i >= self.n * 2:
                continue
            if i >= self.n:
                if left <= i <= right:
                    ans += self.cover[i]
                continue
            if left <= s <= t <= right:
                ans += self.cover[i]
            elif s < t:
                m = s + (t - s) // 2
                if (i<<1) < self.n*2:
                    stack.append((s, m, i << 1))
                if (i << 1)|1 < self.n * 2:
                    stack.append((m + 1, t, (i << 1) | 1))
        return ans
