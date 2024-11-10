from operator import add


class PointUpdateRangeQuery:

    def __init__(self, n, initial=0, merge=add):
        self.n = n
        self.initial = initial
        self.merge = merge
        self.cover = [initial] * 2 * self.n
        return

    def push_up(self, i):
        self.cover[i] = self.merge(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def build(self, nums):
        for i in range(self.n):
            self.cover[i + self.n] = nums[i]
        for i in range(self.n - 1, 0, -1):
            self.push_up(i)
        return

    def get(self):
        return self.cover[self.n:]

    def point_update(self, ind, val):
        assert 0 <= ind < self.n
        ind += self.n
        self.cover[ind] = self.merge(self.cover[ind], val)
        while ind > 1:
            ind //= 2
            self.push_up(ind)
        return

    def range_query(self, left, right):
        assert 0 <= left <= right < self.n
        ans_left = ans_right = self.initial
        left += self.n
        right += self.n + 1
        while left < right:
            if left & 1:
                ans_left = self.merge(ans_left, self.cover[left])
                left += 1
            if right & 1:
                right -= 1
                ans_right = self.merge(self.cover[right], ans_right)
            left >>= 1
            right >>= 1
        return self.merge(ans_left, ans_right)


class PointSetPointAddRangeMerge:

    def __init__(self, n, initial=0, merge=add):
        self.n = n
        self.initial = initial
        self.merge = merge
        self.cover = [initial] * 2 * self.n
        return

    def push_up(self, i):
        self.cover[i] = self.merge(self.cover[i << 1], self.cover[(i << 1) | 1])
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
        assert 0 <= ind < self.n
        ind += self.n
        self.cover[ind] = val
        while ind > 1:
            ind //= 2
            self.push_up(ind)
        return

    def point_add(self, ind, val):
        assert 0 <= ind < self.n
        ind += self.n
        self.cover[ind] += val
        while ind > 1:
            ind //= 2
            self.push_up(ind)
        return

    def range_merge(self, left, right):
        assert 0 <= left <= right < self.n
        ans_left = ans_right = self.initial
        left += self.n
        right += self.n + 1
        while left < right:
            if left & 1:
                ans_left = self.merge(ans_left, self.cover[left])
                left += 1
            if right & 1:
                right -= 1
                ans_right = self.merge(self.cover[right], ans_right)
            left >>= 1
            right >>= 1
        return self.merge(ans_left, ans_right)


class PointSetPointAddRangeSum:

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

    def point_add(self, ind, val):
        ind += self.n
        self.cover[ind] += val
        while ind > 1:
            ind //= 2
            self.push_up(ind)
        return

    def range_sum(self, left, right):
        ans_left = ans_right = 0
        left += self.n
        right += self.n + 1
        while left < right:
            if left & 1:
                ans_left = ans_left + self.cover[left]
                left += 1
            if right & 1:
                right -= 1
                ans_right = self.cover[right] + ans_right
            left >>= 1
            right >>= 1
        return ans_left + ans_right


class RangeMergePointGet:
    def __init__(self, n, initial=0, merge=add):
        self.n = n
        self.merge = merge
        self.initial = initial
        self.cover = [self.initial] * (2 * self.n)
        return

    def build(self, nums):
        for i in range(self.n):
            self.cover[i + self.n] = nums[i]
        return

    def push_up(self, i):
        self.cover[i] = self.merge(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def range_merge(self, left, right, val):
        left += self.n
        right += self.n + 1

        while left < right:
            if left & 1:
                self.cover[left] = self.merge(self.cover[left], val)
                left += 1
            if right & 1:
                right -= 1
                self.cover[right] = self.merge(self.cover[right], val)
            left >>= 1
            right >>= 1
        return

    def get(self):
        for i in range(1, self.n):
            self.cover[i << 1] = self.merge(self.cover[i << 1], self.cover[i])
            self.cover[(i << 1) | 1] = self.merge(self.cover[(i << 1) | 1], self.cover[i])
            self.cover[i] = self.initial
        return self.cover[self.n:]

    def point_get(self, ind):
        ans = self.initial
        ind += self.n
        while ind > 0:
            ans = self.merge(self.cover[ind], ans)
            ind //= 2
        return ans


class RangeAddPointGet:
    def __init__(self, n):
        self.n = n
        self.cover = [0] * (2 * self.n)
        return

    def build(self, nums):
        for i in range(self.n):
            self.cover[i + self.n] = nums[i]
        return

    def push_up(self, i):
        self.cover[i] = self.cover[i << 1] + self.cover[(i << 1) | 1]
        return

    def range_add(self, left, right, val):
        left += self.n
        right += self.n + 1

        while left < right:
            if left & 1:
                self.cover[left] += val
                left += 1
            if right & 1:
                right -= 1
                self.cover[right] += val
            left >>= 1
            right >>= 1
        return

    def get(self):
        for i in range(1, self.n):
            self.cover[i << 1] += self.cover[i]
            self.cover[(i << 1) | 1] += self.cover[i]
            self.cover[i] = 0
        return self.cover[self.n:]

    def point_get(self, ind):
        ans = 0
        ind += self.n
        while ind > 0:
            ans += self.cover[ind]
            ind //= 2
        return ans


class LazySegmentTree:
    def __init__(self, n, combine, cover_initial, merge_cover, merge_tag, tag_initial, num_to_cover):
        self.n = n
        self.combine = combine  # method of cover push_up
        self.cover_initial = cover_initial  # cover_initial value of cover
        self.merge_cover = merge_cover  # method of tag to cover
        self.merge_tag = merge_tag  # method of tag merge
        self.tag_initial = tag_initial  # cover_initial value of tag
        self.num_to_cover = num_to_cover  # cover_initial value from num to cover
        self.lazy_tag = [self.tag_initial] * (2 * self.n)
        self.h = 0
        while (1 << self.h) < n:
            self.h += 1
        self.cover = [self.cover_initial] * (2 * self.n)
        self.cnt = [1] * (2 * self.n)
        for i in range(self.n - 1, 0, -1):
            self.cnt[i] = self.cnt[i << 1] + self.cnt[(i << 1) | 1]
        return

    def build(self, nums):
        for i in range(self.n):
            self.cover[i + self.n] = self.num_to_cover(nums[i])
        for i in range(self.n - 1, 0, -1):
            self.cover[i] = self.combine(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def push_up(self, i):
        while i > 1:
            i >>= 1
            self.cover[i] = self.combine(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def push_down(self, i):
        for s in range(self.h, 0, -1):
            x = i >> s
            if self.lazy_tag[x] != self.tag_initial:
                self.make_tag(x << 1, self.lazy_tag[x])
                self.make_tag((x << 1) | 1, self.lazy_tag[x])
                self.lazy_tag[x] = self.tag_initial
        x = i
        if self.lazy_tag[x] != self.tag_initial:
            if (i << 1) < self.n * 2:
                self.make_tag(x << 1, self.lazy_tag[x])
            if ((i << 1) | 1) < self.n * 2:
                self.make_tag((x << 1) | 1, self.lazy_tag[x])
            self.lazy_tag[x] = self.tag_initial
        return

    def make_tag(self, i, val):
        self.cover[i] = self.merge_cover(self.cover[i], val, self.cnt[i])  # cover val length
        self.lazy_tag[i] = self.merge_tag(val, self.lazy_tag[i])
        return

    def range_update(self, left, right, val):
        left += self.n
        right += self.n + 1
        ll = left
        rr = right
        self.push_down(ll)
        self.push_down(rr - 1)
        while left < right:
            if left & 1:
                self.make_tag(left, val)
                left += 1
            if right & 1:
                right -= 1
                self.make_tag(right, val)
            left >>= 1
            right >>= 1
        self.push_down(ll)
        self.push_down(rr - 1)
        self.push_up(ll)
        self.push_up(rr - 1)
        return

    def get(self):
        for i in range(1, self.n):
            if self.lazy_tag[i] != self.tag_initial:
                self.make_tag(i << 1, self.lazy_tag[i])
                self.make_tag((i << 1) | 1, self.lazy_tag[i])
                self.lazy_tag[i] = self.tag_initial
        return self.cover[self.n:]

    def point_get(self, ind):
        ans = 0
        ind += self.n
        while ind > 0:
            ans += self.lazy_tag[ind]
            ind //= 2
        return ans

    def range_query(self, left, right):
        ans_left = ans_right = self.cover_initial
        left += self.n
        right += self.n + 1
        self.push_down(left)
        self.push_down(right - 1)
        while left < right:
            if left & 1:
                ans_left = self.combine(ans_left, self.cover[left])
                left += 1
            if right & 1:
                right -= 1
                ans_right = self.combine(self.cover[right], ans_right)
            left >>= 1
            right >>= 1
        return self.combine(ans_left, ans_right)


class LazySegmentTreeLength:
    def __init__(self, n, combine_cover, cover_initial, merge_cover_tag, merge_tag_tag, tag_initial):
        self.n = n
        self.cover_initial = cover_initial
        self.merge_cover_tag = merge_cover_tag
        self.merge_tag_tag = merge_tag_tag
        self.tag_initial = tag_initial
        self.lazy_tag = [self.tag_initial] * (2 * self.n)
        self.h = 0
        while (1 << self.h) < n:
            self.h += 1
        self.combine_cover = combine_cover
        self.cover = [self.cover_initial] * (2 * self.n)
        return

    def build(self, nums):
        for i in range(self.n):
            if nums[i]:
                self.cover[i + self.n] = (1, 1, 0, 1, 1, 1)
            else:
                self.cover[i + self.n] = (-1, -1, 1, 0, 1, 0)
        for i in range(self.n - 1, 0, -1):
            self.cover[i] = self.combine_cover(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def push_up(self, i):
        while i > 1:
            i >>= 1
            self.cover[i] = self.combine_cover(self.cover[i << 1], self.cover[(i << 1) | 1])
        return

    def push_down(self, i):
        for s in range(self.h, 0, -1):
            x = i >> s
            if self.lazy_tag[x] != self.tag_initial:
                self.make_tag(x << 1, self.lazy_tag[x])
                self.make_tag((x << 1) | 1, self.lazy_tag[x])
                self.lazy_tag[x] = self.tag_initial
        x = i
        if self.lazy_tag[x] != self.tag_initial:
            if (i << 1) < self.n * 2:
                self.make_tag(x << 1, self.lazy_tag[x])
            if ((i << 1) | 1) < self.n * 2:
                self.make_tag((x << 1) | 1, self.lazy_tag[x])
            self.lazy_tag[x] = self.tag_initial
        return

    def make_tag(self, i, val):
        self.cover[i] = self.merge_cover_tag(self.cover[i], val)
        self.lazy_tag[i] = self.merge_tag_tag(val, self.lazy_tag[i])
        return

    def range_update(self, left, right, val):
        left += self.n
        right += self.n + 1
        ll = left
        rr = right
        self.push_down(ll)
        self.push_down(rr - 1)
        while left < right:
            if left & 1:
                self.make_tag(left, val)
                left += 1
            if right & 1:
                right -= 1
                self.make_tag(right, val)
            left >>= 1
            right >>= 1
        self.push_down(ll)
        self.push_down(rr - 1)

        self.push_up(ll)
        self.push_up(rr - 1)
        return

    def get(self):
        for i in range(1, self.n):
            if self.lazy_tag[i] != self.tag_initial:
                self.make_tag(i << 1, self.lazy_tag[i])
                self.make_tag((i << 1) | 1, self.lazy_tag[i])
                self.lazy_tag[i] = self.tag_initial
        return self.cover[self.n:]

    def point_get(self, ind):
        ans = 0
        ind += self.n
        while ind > 0:
            ans += self.lazy_tag[ind]
            ind //= 2
        return ans

    def range_query(self, left, right):
        ans_left = ans_right = self.cover_initial
        left += self.n
        right += self.n + 1
        self.push_down(left)
        self.push_down(right - 1)
        while left < right:
            if left & 1:
                ans_left = self.combine_cover(ans_left, self.cover[left])
                left += 1
            if right & 1:
                right -= 1
                ans_right = self.combine_cover(self.cover[right], ans_right)
            left >>= 1
            right >>= 1
        return self.combine_cover(ans_left, ans_right)
