

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
