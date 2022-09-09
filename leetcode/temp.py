
from functools import lru_cache

from sortedcontainers import SortedList, SortedDict, SortedSet

from sortedcontainers import SortedDict


st = """执行用时：
164 ms
, 在所有 Python3 提交中击败了
100.00%
的用户
内存消耗：
16 MB
, 在所有 Python3 提交中击败了
100.00%
的用户
通过测试用例：
95 / 95"""
st = st.split("\n")
i, j = st.index('内存消耗：'), st.index('通过测试用例：')
st[i-2] = " " + st[i-2] + " "
st[j-2] = " " + st[j-2] + " "
print("- " + "".join(st[:i]))
print("- " + "".join(st[i:j]))
print("- " + "".join(st[j:]))
import math
def dfs(n):
    if n <= 2:
        return n
    k = math.floor(math.sqrt(2*n+1/4)-1/2)
    cur = k*(k+1)//2
    return n+dfs(k*(k-1)//2+max(0, n-cur-1))

print(dfs(5))