

from collections import defaultdict


class SegmentTreeRangeSum:
    def __init__(self):
        self.cover = defaultdict(int)
        self.lazy = defaultdict(int)
        self.count = 0

    def push_down(self, i):
        if self.lazy[i]:
            self.cover[2 * i] = self.lazy[i]
            self.cover[2 * i + 1] = self.lazy[i]

            self.lazy[2 * i] = self.lazy[i]
            self.lazy[2 * i + 1] = self.lazy[i]

            self.lazy[i] = 0

    def update(self, left, r, s, t, val, i):
        if left <= s and t <= r:
            self.cover[i] += val
            self.lazy[i] = val
            return

        self.push_down(i)
        m = s + (t - s) // 2
        if left <= m:
            self.update(left, r, s, m, val, 2 * i)
        if r > m:
            self.update(left, r, m + 1, t, val, 2 * i + 1)
        self.cover[i] = self.cover[2 * i] + self.cover[2 * i + 1]
        return

    def query(self, left, r, s, t, i):
        if left <= s and t <= r:
            return self.cover[i]
        self.push_down(i)
        m = s + (t - s) // 2
        ans = 0
        if left <= m:
            ans += self.query(left, r, s, m, 2 * i)
        if r > m:
            ans += self.query(left, r, m + 1, t, 2 * i + 1)
        return ans


class SegmentTreeRangeMax:
    def __init__(self):
        self.height = defaultdict(int)
        self.lazy = defaultdict(int)

    def push_down(self, i):
        # 懒标记下放，注意取最大值
        if self.lazy[i]:
            self.height[2*i] = self.height[2*i] if self.height[2*i] > self.lazy[i] else self.lazy[i]
            self.height[2*i+1] = self.height[2*i+1] if self.height[2*i+1] > self.lazy[i] else self.lazy[i]

            self.lazy[2*i] = self.lazy[2*i] if self.lazy[2*i] > self.lazy[i] else self.lazy[i]
            self.lazy[2*i+1] = self.lazy[2*i+1] if self.lazy[2*i+1] > self.lazy[i] else self.lazy[i]

            self.lazy[i] = 0
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最大值
        if l<=s and t<=r:
            self.height[i] = self.height[i] if self.height[i] > val else val
            self.lazy[i] = self.lazy[i] if self.lazy[i] > val else val
            return
        self.push_down(i)
        m = s+(t-s)//2
        if l<=m: # 注意左右子树的边界与范围
            self.update(l, r, s, m, val, 2*i)
        if r>m:
            self.update(l, r, m+1, t, val, 2*i+1)
        self.height[i] = self.height[2*i] if self.height[2*i] > self.height[2*i+1] else self.height[2*i+1]
        return

    def query(self, l, r, s, t, i):
        # 查询区间的最大值
        if l<=s and t<=r:
            return self.height[i]
        self.push_down(i)
        m = s+(t-s)//2
        highest = 0
        if l<=m:
            cur = self.query(l, r, s, m, 2*i)
            if cur > highest:
                highest = cur
        if r>m:
            cur = self.query(l, r, m+1, t, 2*i+1)
            if cur > highest:
                highest = cur
        return highest

# 作者：liupengsay
# 链接：https://leetcode.cn/problems/the-skyline-problem/solution/by-liupengsay-isfo/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


class SegmentTreeMin:
    def __init__(self):
        self.height = defaultdict(lambda: float("inf"))
        self.lazy = defaultdict(lambda: float("inf"))

    def push_down(self, i):
        # 懒标记下放，注意取最小值
        if self.lazy[i] < float("inf"):
            self.height[2*i] = self.height[2*i] if self.height[2*i] < self.lazy[i] else self.lazy[i]
            self.height[2*i+1] = self.height[2*i+1] if self.height[2*i+1] < self.lazy[i] else self.lazy[i]

            self.lazy[2*i] = self.lazy[2*i] if self.lazy[2*i] < self.lazy[i] else self.lazy[i]
            self.lazy[2*i+1] = self.lazy[2*i+1] if self.lazy[2*i+1] < self.lazy[i] else self.lazy[i]

            self.lazy[i] = float("inf")
        return

    def update(self, l, r, s, t, val, i):
        # 更新区间最小值
        if l<=s and t<=r:
            self.height[i] = self.height[i] if self.height[i] < val else val
            self.lazy[i] = self.lazy[i] if self.lazy[i] < val else val
            return
        self.push_down(i)
        m = s+(t-s)//2
        if l<=m: # 注意左右子树的边界与范围
            self.update(l, r, s, m, val, 2*i)
        if r>m:
            self.update(l, r, m+1, t, val, 2*i+1)
        self.height[i] = self.height[2*i] if self.height[2*i] < self.height[2*i+1] else self.height[2*i+1]
        return

    def query(self, l, r, s, t, i):
        # 查询区间的最小值
        if l<=s and t<=r:
            return self.height[i]
        self.push_down(i)
        m = s+(t-s)//2
        highest = float("inf")
        if l<=m:
            cur = self.query(l, r, s, m, 2*i)
            if cur < highest:
                highest = cur
        if r>m:
            cur = self.query(l, r, m+1, t, 2*i+1)
            if cur < highest:
                highest = cur
        return highest if highest < float("inf") else -1


def test_segment_tree_range_sum():

    def check(nums):
        low = min(nums)
        nums = [num - low + 1 for num in nums]
        ceil = max(nums)

        segment_tree = SegmentTreeRangeSum()
        ans = 0
        for num in nums:
            left = segment_tree.query(0, num - 1, 0, ceil + 1, 1)
            right = segment_tree.query(num + 1, ceil, 0, ceil + 1, 1)
            segment_tree.update(num, num, 0, ceil + 1, 1, 1)
            ans += left if left < right else right
        return ans

    assert check(nums=[1, 3, 3, 3, 2, 4, 2, 1, 2]) == 4
    assert check(nums=[1, 2, 3, 6, 5, 4]) == 3
    assert check(nums=[1, 5, 6, 2]) == 1
    return


if __name__ == '__main__':
    test_segment_tree_range_sum()
