

```Python3
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        
        '''KMP模板'''
        def prefix_function(s):     
            n = len(s)
            pi = [0] * n

            j = 0
            for i in range(1, n):
                while j>0 and s[i] != s[j]:     # 当前位置s[i]与s[j]不等
                    j = pi[j-1]                 # j指向之前位置，s[i]与s[j]继续比较

                if s[i] == s[j]:                # s[i]与s[j]相等，j+1，指向后一位
                    j += 1
                
                pi[i] = j
            return pi
        

        '''主程序'''
        pi = prefix_function(s+'#'+s[::-1])     # s+'#'+s[n-1,...,0]的前缀函数
        if pi[-1] == len(s):                    # 前缀函数的最后一位即为s的最长回文前缀的长度
            return s
        else:
            return s[pi[-1]:][::-1] + s

作者：flix
链接：https://leetcode.cn/problems/shortest-palindrome/solution/by-flix-be4y/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


```python3
# 计算s[i:]与s的最长公共前缀
# Python Version
def z_function(s):
    n = len(s)
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i <= r and z[i - l] < r - i + 1:
            z[i] = z[i - l]
        else:
            z[i] = max(0, r - i + 1)
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1
    return z


# Python Version
# 计算s[:i]与s[:i]的最长公共真前缀与真后缀
def prefix_function(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi
```