"""


"""
"""
算法：二分查找、有序集合
功能：利用单调性确定最优选择，通常可以使用SortedList
题目：xx（xx）
L2468 根据限制分割消息（https://leetcode.cn/problems/split-message-based-on-limit/）根据长度限制进行二分
L2426 满足不等式的数对数目（https://leetcode.cn/problems/number-of-pairs-satisfying-inequality/）根据不等式变换和有序集合进行二分查找
L2179 统计数组中好三元组数目（https://leetcode.cn/problems/count-good-triplets-in-an-array/）维护区间范围内的个数
L2141 同时运行 N 台电脑的最长时间（https://leetcode.cn/problems/maximum-running-time-of-n-computers/）贪心选择最大的 N 个电池作为基底，然后二分确定在其余电池的加持下可以运行的最长时间
L2102 序列顺序查询（https://leetcode.cn/problems/sequentially-ordinal-rank-tracker/）使用有序集合维护优先级姓名实时查询
参考：OI WiKi（xx）
"""

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy

import bisect
import random
import re
import unittest

from typing import List
import heapq
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
from itertools import combinations
from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict
from functools import reduce
from operator import xor
from functools import lru_cache

import random
from itertools import permutations, combinations
import numpy as np

from decimal import Decimal

import heapq
import copy


class LocalSortedList:
    def __init__(self, iterable=[], _load=200):
        """Initialize sorted list instance."""
        values = sorted(iterable)
        self._len = _len = len(values)
        self._load = _load
        self._lists = _lists = [values[i:i + _load] for i in range(0, _len, _load)]
        self._list_lens = [len(_list) for _list in _lists]
        self._mins = [_list[0] for _list in _lists]
        self._fen_tree = []
        self._rebuild = True

    def _fen_build(self):
        """Build a fenwick tree instance."""
        self._fen_tree[:] = self._list_lens
        _fen_tree = self._fen_tree
        for i in range(len(_fen_tree)):
            if i | i + 1 < len(_fen_tree):
                _fen_tree[i | i + 1] += _fen_tree[i]
        self._rebuild = False

    def _fen_update(self, index, value):
        """Update `fen_tree[index] += value`."""
        if not self._rebuild:
            _fen_tree = self._fen_tree
            while index < len(_fen_tree):
                _fen_tree[index] += value
                index |= index + 1

    def _fen_query(self, end):
        """Return `sum(_fen_tree[:end])`."""
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        x = 0
        while end:
            x += _fen_tree[end - 1]
            end &= end - 1
        return x

    def _fen_findkth(self, k):
        """Return a pair of (the largest `idx` such that `sum(_fen_tree[:idx]) <= k`, `k - sum(_fen_tree[:idx])`)."""
        _list_lens = self._list_lens
        if k < _list_lens[0]:
            return 0, k
        if k >= self._len - _list_lens[-1]:
            return len(_list_lens) - 1, k + _list_lens[-1] - self._len
        if self._rebuild:
            self._fen_build()

        _fen_tree = self._fen_tree
        idx = -1
        for d in reversed(range(len(_fen_tree).bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < len(_fen_tree) and k >= _fen_tree[right_idx]:
                idx = right_idx
                k -= _fen_tree[idx]
        return idx + 1, k

    def _delete(self, pos, idx):
        """Delete value at the given `(pos, idx)`."""
        _lists = self._lists
        _mins = self._mins
        _list_lens = self._list_lens

        self._len -= 1
        self._fen_update(pos, -1)
        del _lists[pos][idx]
        _list_lens[pos] -= 1

        if _list_lens[pos]:
            _mins[pos] = _lists[pos][0]
        else:
            del _lists[pos]
            del _list_lens[pos]
            del _mins[pos]
            self._rebuild = True

    def _loc_left(self, value):
        """Return an index pair that corresponds to the first position of `value` in the sorted list."""
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._mins

        lo, pos = -1, len(_lists) - 1
        while lo + 1 < pos:
            mi = (lo + pos) >> 1
            if value <= _mins[mi]:
                pos = mi
            else:
                lo = mi

        if pos and value <= _lists[pos - 1][-1]:
            pos -= 1

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value <= _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def _loc_right(self, value):
        """Return an index pair that corresponds to the last position of `value` in the sorted list."""
        if not self._len:
            return 0, 0

        _lists = self._lists
        _mins = self._mins

        pos, hi = 0, len(_lists)
        while pos + 1 < hi:
            mi = (pos + hi) >> 1
            if value < _mins[mi]:
                hi = mi
            else:
                pos = mi

        _list = _lists[pos]
        lo, idx = -1, len(_list)
        while lo + 1 < idx:
            mi = (lo + idx) >> 1
            if value < _list[mi]:
                idx = mi
            else:
                lo = mi

        return pos, idx

    def add(self, value):
        """Add `value` to sorted list."""
        _load = self._load
        _lists = self._lists
        _mins = self._mins
        _list_lens = self._list_lens

        self._len += 1
        if _lists:
            pos, idx = self._loc_right(value)
            self._fen_update(pos, 1)
            _list = _lists[pos]
            _list.insert(idx, value)
            _list_lens[pos] += 1
            _mins[pos] = _list[0]
            if _load + _load < len(_list):
                _lists.insert(pos + 1, _list[_load:])
                _list_lens.insert(pos + 1, len(_list) - _load)
                _mins.insert(pos + 1, _list[_load])
                _list_lens[pos] = _load
                del _list[_load:]
                self._rebuild = True
        else:
            _lists.append([value])
            _mins.append(value)
            _list_lens.append(1)
            self._rebuild = True

    def discard(self, value):
        """Remove `value` from sorted list if it is a member."""
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_right(value)
            if idx and _lists[pos][idx - 1] == value:
                self._delete(pos, idx - 1)

    def remove(self, value):
        """Remove `value` from sorted list; `value` must be a member."""
        _len = self._len
        self.discard(value)
        if _len == self._len:
            raise ValueError('{0!r} not in list'.format(value))

    def pop(self, index=-1):
        """Remove and return value at `index` in sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        value = self._lists[pos][idx]
        self._delete(pos, idx)
        return value

    def bisect_left(self, value):
        """Return the first index to insert `value` in the sorted list."""
        pos, idx = self._loc_left(value)
        return self._fen_query(pos) + idx

    def bisect_right(self, value):
        """Return the last index to insert `value` in the sorted list."""
        pos, idx = self._loc_right(value)
        return self._fen_query(pos) + idx

    def count(self, value):
        """Return number of occurrences of `value` in the sorted list."""
        return self.bisect_right(value) - self.bisect_left(value)

    def __len__(self):
        """Return the size of the sorted list."""
        return self._len

    def __getitem__(self, index):
        """Lookup value at `index` in sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        return self._lists[pos][idx]

    def __delitem__(self, index):
        """Remove value at `index` from sorted list."""
        pos, idx = self._fen_findkth(self._len + index if index < 0 else index)
        self._delete(pos, idx)

    def __contains__(self, value):
        """Return true if `value` is an element of the sorted list."""
        _lists = self._lists
        if _lists:
            pos, idx = self._loc_left(value)
            return idx < len(_lists[pos]) and _lists[pos][idx] == value
        return False

    def __iter__(self):
        """Return an iterator over the sorted list."""
        return (value for _list in self._lists for value in _list)

    def __reversed__(self):
        """Return a reverse iterator over the sorted list."""
        return (value for _list in reversed(self._lists) for value in reversed(_list))

    def __repr__(self):
        """Return string representation of sorted list."""
        return 'SortedList({0})'.format(list(self))


class DefineSortedList:
    def __init__(self, lst=[]):
        # 也可以考虑使用bisect实现
        self.lst = lst
        self.lst.sort()
        return

    def add(self, num):
        if not self.lst:
            self.lst.append(num)
            return
        if self.lst[-1] <= num:
            self.lst.append(num)
            return
        if self.lst[0] >= num:
            self.lst.insert(0, num)
            return
        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] >= num:
                j = mid
            else:
                i = mid
        if self.lst[i] >= num:
            self.lst.insert(i, num)
        else:
            self.lst.insert(j, num)
        return

    def discard(self, num):
        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] > num:
                j = mid
            elif self.lst[mid] < num:
                i = mid
            else:
                self.lst.pop(mid)
                return
        if self.lst[i] == num:
            self.lst.pop(i)
        elif self.lst[j] == num:
            self.lst.pop(j)
        return

    def bisect_left(self, num):
        if not self.lst:
            return 0
        if self.lst[-1] < num:
            return len(self.lst)
        if self.lst[0] > num:
            return 0
        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] >= num:
                j = mid
            else:
                i = mid
        if self.lst[i] >= num:
            return i
        return j

    def bisect_right(self, num):
        if not self.lst:
            return 0
        if self.lst[-1] <= num:
            return len(self.lst)
        if self.lst[0] > num:
            return 0

        i = 0
        j = len(self.lst) - 1
        while i < j - 1:
            mid = i + (j - i) // 2
            if self.lst[mid] <= num:
                i = mid
            else:
                j = mid
        if self.lst[j] >= num:
            return j
        return i


class TestGeneral(unittest.TestCase):

    def test_define_sorted_list(self):
        for _ in range(10):
            floor = -10**8
            ceil = 10**8
            low = -5*10**7
            high = 6*10**8
            n = 10**4
            # add
            lst = SortedList()
            define = DefineSortedList([])
            local_lst = LocalSortedList()
            for _ in range(n):
                num = random.randint(low, high)
                lst.add(num)
                define.add(num)
                local_lst.add(num)
            assert all(lst[i] == define.lst[i] == local_lst[i] for i in range(n))
            # discard
            for _ in range(n):
                num = random.randint(low, high)
                lst.discard(num)
                define.discard(num)
                local_lst.discard(num)
            m = len(lst)
            assert all(lst[i] == define.lst[i] == local_lst[i] for i in range(m))
            # bisect_left
            for _ in range(n):
                num = random.randint(low, high)
                lst.add(num)
                define.add(num)
                local_lst.add(num)
            for _ in range(n):
                num = random.randint(floor, ceil)
                assert lst.bisect_left(num) == define.bisect_left(num) == local_lst.bisect_left(num)
            # bisect_right
            for _ in range(n):
                num = random.randint(floor, ceil)
                assert lst.bisect_right(num) == define.bisect_right(num) == local_lst.bisect_right(num)
        return


if __name__ == '__main__':
    unittest.main()

