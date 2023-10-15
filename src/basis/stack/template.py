

class MaxStack:
    # 模板：经典最大栈
    def __init__(self):
        return

    def gen_result(self):
        return


class MinStack:
    # 模板：经典最小栈
    def __init__(self):
        return

    def gen_result(self):
        return

    @staticmethod
    def ac_128(ac=FastIO()):
        # 模板：经典栈与指针模拟
        pre = []
        post = []
        pre_sum = [0]
        pre_ceil = [-inf]
        for _ in range(ac.read_int()):
            lst = ac.read_list_strs()
            if lst[0] == "I":
                pre.append(int(lst[1]))
                pre_sum.append(pre_sum[-1]+int(lst[1]))
                pre_ceil.append(ac.max(pre_ceil[-1], pre_sum[-1]))
            elif lst[0] == "D":
                if pre:
                    pre.pop()
                    pre_sum.pop()
                    pre_ceil.pop()
            elif lst[0] == "L":
                if pre:
                    post.append(pre.pop())
                    pre_sum.pop()
                    pre_ceil.pop()
            elif lst[0] == "R":
                if post:
                    x = post.pop()
                    pre.append(x)
                    pre_sum.append(pre_sum[-1] + x)
                    pre_ceil.append(ac.max(pre_ceil[-1], pre_sum[-1]))
            else:
                ac.st(pre_ceil[int(lst[1])])
        return

    @staticmethod
    def ac_129_1(ac=FastIO()):
        # 模板：经典卡特兰数，栈模拟判定出栈入栈合法性
        n = ac.read_int()
        m = ac.min(5, n)

        pre = list(range(1, n+1))

        def check(lst):
            lst = deque(lst)
            stack = []
            for num in pre:
                stack.append(num)
                while stack and stack[-1] == lst[0]:
                    stack.pop()
                    lst.popleft()
            return not stack

        cnt = 0
        for item in permutations(list(range(n-m+1, n+1)), m):
            cur = list(range(1, n-m+1))+list(item)
            if check(cur):
                ac.st("".join(str(x) for x in cur))
                cnt += 1
            if cnt == 20:
                break
        return

    @staticmethod
    def ac_129_2(ac=FastIO()):
        # 模板：使用回溯模拟出栈入栈所有可能的排列

        def dfs(i):
            nonlocal cnt, post, pre
            if cnt >= 20:
                return
            if i == n:
                cnt += 1
                ac.st("".join(str(x) for x in res))
                return

            if pre:
                res.append(pre.pop())
                dfs(i+1)
                pre.append(res.pop())

            if post:
                pre.append(post.popleft())
                dfs(i)
                post.appendleft(pre.pop())
            return

        n = ac.read_int()
        post = deque(list(range(1, n+1)))
        res = []
        pre = []
        cnt = 0
        dfs(0)
        return

    @staticmethod
    def ac_129_3(ac=FastIO()):
        # 模板：使用迭代写法替换深搜与回溯
        n = ac.read_int()
        cnt = 0
        stack = [[[], [], 0]]
        while stack and cnt < 20:
            pre, res, ind = stack.pop()
            if len(res) == n:
                cnt += 1
                ac.st("".join(str(x) for x in res))
            else:
                if ind + 1 <= n:
                    stack.append([pre+[ind+1], res[:], ind+1])
                if pre:
                    stack.append([pre[:-1], res+[pre[-1]], ind])
        return

    @staticmethod
    def lg_p1974(ac=FastIO()):
        # 模板：贪心队列模拟
        n = ac.read_int()
        stack = deque([1] * n)
        while len(stack) >= 2:
            a, b = stack.popleft(), stack.popleft()
            stack.append(a * b + 1)
        ac.st(stack[0])
        return

    @staticmethod
    def lg_p3719(ac=FastIO()):
        # 模板：栈模拟
        s = ac.read_str()
        stack = []
        for w in s:
            if w != ")":
                stack.append(w)
            else:
                pre = ""
                while stack and stack[-1] != "(":
                    w = stack.pop()
                    pre += w
                stack.pop()
                x = max(len(t) for t in pre.split("|"))
                stack.append("a" * x)

        pre = "".join(stack)
        x = max(len(t) for t in pre.split("|"))
        ac.st(x)
        return

    @staticmethod
    def ac_4865(ac=FastIO()):
        # 模板：经典栈模拟
        m = ac.read_int()
        lst = ac.read_list_strs()
        n = len(lst)
        if n != m*2-1:
            ac.st("Error occurred")
            return
        if m == 1:
            ac.st("int")
            return

        stack = []
        for i in range(n):
            if lst[i] == "int":
                stack.append([[i, i], "int"])
                # 维护每个函数段的左右边界
                while len(stack) >= 3 and [ls[1] for ls in stack[-3:]] == ["pair", "int", "int"]:
                    lst[stack[-1][0][1]] += ">"
                    lst[stack[-2][0][1]] += ","
                    lst[stack[-2][0][0]] = "<" + lst[stack[-2][0][0]]
                    stack[-3][0][1] = stack[-1][0][1]
                    stack[-3][0][0] = stack[-3][0][0]
                    stack[-3][1] = "int"
                    stack.pop()
                    stack.pop()
            else:
                stack.append([[i, i], "pair"])

        if len(stack) > 1:
            ac.st("Error occurred")
            return
        ac.st("".join(lst))
        return

    @staticmethod
    def ac_5136(ac=FastIO()):
        # 模板：经典栈倒序模拟
        s = ac.read_str()
        n = len(s)
        ans = [0]*n
        right = 0
        post = deque()
        for i in range(n-1, -1, -1):
            if s[i] == "#":
                post.append(i)
            elif s[i] == "(":
                if right:
                    right -= 1
                else:
                    if not post:
                        ac.st(-1)
                        return
                    ans[post[0]] += 1
            else:
                right += 1
            while len(post) >= 2 and ans[post[0]]:
                post.popleft()
        while post and ans[post[0]]:
            post.popleft()
        if post or right:
            ac.st(-1)
            return
        for i in range(n):
            if s[i] == "#":
                ac.st(ans[i])
        return


