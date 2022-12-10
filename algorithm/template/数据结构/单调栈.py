class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:

        def check(lst):
            post = [n - 1] * n
            stack = []
            for x in range(n):
                while stack and lst[stack[-1]] > lst[x]:
                    post[stack.pop()] = x - 1
                stack.append(x)

            prev = [0] * n
            stack = []
            for x in range(n - 1, -1, -1):
                while stack and lst[stack[-1]] > lst[x]:
                    prev[stack.pop()] = x + 1
                stack.append(x)

            res = 0
            for x in range(n):
                if lst[x] * (post[x] - prev[x] + 1) > res:
                    res = lst[x] * (post[x] - prev[x] + 1)
            return res

        ans =  0
        m, n = len(matrix), len(matrix[0])
        pre = [0] * n
        for i in range(m):
            cur = [0] * n
            for j in range(n):
                if matrix[i][j] == "1":
                    cur[j] = pre[j] + 1
            pre = cur
            area = check(pre)
            if area > ans:
                ans = area
        return ans


MOD = 10**9 + 7
class Solution:
    def maxSumMinProduct(self, nums: List[int]) -> int:
        n = len(nums)

        post = [n-1]*n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                post[stack.pop()] = i-1
            stack.append(i)

        pre = [0]*n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                pre[stack.pop()] = i+1
            stack.append(i)

        pre_fix = [0]
        for num in nums:
            pre_fix.append(pre_fix[-1]+num)

        ans = 0
        for i in range(n):
            if nums[i]*(post[i]-pre[i]+1) > ans:
                ans = nums[i]*(post[i]-pre[i]+1)
        return ans % MOD

"""

#求长方形个数题目：https://www.luogu.com.cn/problem/P1950
import sys

input = lambda: sys.stdin.readline()
print = lambda x: sys.stdout.write(str(x)+'\n')


m, n = [int(w) for w in input().strip().split() if w]
ans = 0
pre = [0]*n
for _ in range(m):
    s = input().strip()
    for j in range(n):
        if s[j] == ".":
            pre[j] += 1
        else:
            pre[j] = 0

    right = [n-1]*n
    left = [0] * n
    stack = []
    for j in range(n):
        while stack and pre[stack[-1]] >= pre[j]:
            right[stack.pop()] = j-1
        if stack:
            left[j] = stack[-1]+1
        stack.append(j)

    ans += sum((right[j]-j+1)*(j-left[j]+1)*pre[j] for j in range(n))
print(ans)

"""
