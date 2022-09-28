

# 统计无重复数字的整数
```Python3

class Solution:
    def countSpecialNumbers(self, n: int) -> int:

        digits = [int(x) for x in str(n+1)]
        ans = 0
        def comb(m,k):
            return 1 if k == 0 else comb(m,k - 1) * (m - k + 1)
        
        for i in range(1, len(digits)):
            ans += 9 * comb(9,i - 1)

        pre = set()
        for st,num in enumerate(digits):
            visit = sum(i not in pre for i in range(0 if st else 1,num))
            ans += visit * comb(10 - st - 1,len(digits) - st - 1)
            if num in pre:
                break
            pre.add(num)
        return ans
```


![数位DP.png](../../picture/数位DP.png)

附：力扣上的数位 DP 题目
233. 数字 1 的个数
     面试题 17.06. 2出现的次数
600. 不含连续1的非负整数
902. 最大为 N 的数字组合
1012. 至少有 1 位重复的数字
1067. 范围内的数字计数
1397. 找到所有好字符串
      Python3JavaC++Go

```Python3
from functools import lru_cache


class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)
        @lru_cache(None)
        def f(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:  # 可以跳过当前数位
                res = f(i + 1, mask, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(0 if is_num else 1, up + 1):  # 枚举要填入的数字 d
                if mask >> d & 1 == 0:  # d 不在 mask 中
                    res += f(i + 1, mask | (1 << d), is_limit and d == up, True)
            return res
        return f(0, 0, True, False)

# 作者：endlesscheng
# 链接：https://leetcode.cn/problems/count-special-integers/solution/shu-wei-dp-mo-ban-by-endlesscheng-xtgx/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

# 1067. 范围内的数字计数
```Python3
class Solution:
    def digitsCount(self, d: int, low: int, high: int) -> int:

        def check(num):
            @lru_cache(None)
            def dfs(i, cnt, is_limit, is_num):
                if i == n:
                    if is_num:
                        return cnt
                    return 0
                res = 0
                if not is_num:
                    res += dfs(i + 1, 0, False, False)

                floor = 0 if is_num else 1
                ceil = int(s[i]) if is_limit else 9
                for x in range(floor, ceil + 1):
                    res += dfs(i + 1, cnt + int(x == d),
                               is_limit and ceil == x, True)
                return res
            s = str(num)
            n = len(s)
            return dfs(0, 0, True, False)
        return check(high) - check(low - 1)
```
