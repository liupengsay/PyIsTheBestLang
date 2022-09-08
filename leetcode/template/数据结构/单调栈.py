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