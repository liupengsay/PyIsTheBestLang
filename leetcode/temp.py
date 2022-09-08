
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict


st = """执行用时：
124 ms
, 在所有 Python3 提交中击败了
28.35%
的用户
内存消耗：
15.7 MB
, 在所有 Python3 提交中击败了
80.31%
的用户
通过测试用例：
64 / 64"""
st = st.split("\n")
i, j = st.index('内存消耗：'), st.index('通过测试用例：')
st[i-2] = " " + st[i-2] + " "
st[j-2] = " " + st[j-2] + " "
print("- " + "".join(st[:i]))
print("- " + "".join(st[i:j]))
print("- " + "".join(st[j:]))