
import unittest


"""

"""
"""
算法：中缀、后缀表达式
功能：xxx
题目：
P1175 表达式的转换（https://www.luogu.com.cn/problem/P1175）
1597. 根据中缀表达式构造二叉表达式树（https://leetcode.cn/problems/build-binary-expression-tree-from-infix-expression/）

参考：OI WiKi（xx）
"""


class Node(object):
    def __init__(self, val=" ", left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeExpression:
    def __init__(self):
        return

    def exp_tree(self, s: str) -> 'Node':
        # 只剩数字的情况
        try:
            int(s)
            return Node(s)
        except ValueError as _:
            pass

        # 处理-(的情形
        if len(s) >= 2 and s[0] == "-" and s[1] == "(":
            cnt = 0
            for i, w in enumerate(s):
                if w == "(":
                    cnt += 1
                elif w == ")":
                    cnt -= 1
                    if not cnt:
                        pre = s[1:i].replace("+", "-").replace("-", "+")
                        s = pre + s[i:]
                        break

        # 处理包含负数的情形
        neg = ""
        if s[0] == "-":
            neg = "-"
            s = s[1:]

        n = len(s)
        cnt = 0
        # 按照运算符号的优先级倒序遍历字符串
        for i in range(n - 1, -1, -1):
            cnt += int(s[i] == ')') - int(s[i] == '(')
            if s[i] in ['+', '-'] and not cnt:
                return Node(s[i], self.exp_tree(neg + s[:i]), self.exp_tree(s[i + 1:]))

        # 注意是从后往前
        for i in range(n - 1, -1, -1):
            cnt += int(s[i] == ')') - int(s[i] == '(')
            if s[i] in ['*', '/'] and not cnt:
                return Node(s[i], self.exp_tree(neg + s[:i]), self.exp_tree(s[i + 1:]))

        # 注意是从前往后
        for i in range(n):
            cnt += int(s[i] == ')') - int(s[i] == '(')
            if s[i] in ['^'] and not cnt:  # 这里的 ^ 表示幂
                return Node(s[i], self.exp_tree(neg + s[:i]), self.exp_tree(s[i + 1:]))

        # 其余则是开头结尾为括号的情况
        return self.exp_tree(s[1:-1])

    def main_1175(self, s):

        # 按照前序、中序与后序变成前缀中缀与后缀表达式
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            dfs(node.right)
            pre.append(node.val)
            return

        ans = []
        root = self.exp_tree(s)
        pre = []
        dfs(root)
        while len(pre) > 1:
            ans.append(pre)
            n = len(pre)
            stack = []
            for i in range(n):
                if pre[i] in "+-*/^":
                    op = pre[i]
                    b = stack.pop()
                    a = stack.pop()
                    op = "//" if op == "/" else op
                    op = "**" if op == "^" else op
                    stack.append(str(eval(f"{a}{op}{b}")))
                    stack += pre[i + 1:]
                    break
                else:
                    stack.append(pre[i])
            pre = stack[:]
        ans.append(pre)
        return ans


class TestGeneral(unittest.TestCase):

    def test_tree_expression(self):
        te = TreeExpression()
        lst = ["2*3^4^2+(5/2-2)", "-2+3", "2*(-5/2+3*2)-33", "((-2+3)*3+5-7/2)^2", "2*(-3)", "0-(-3)", "-(-3)+2"]
        for s in lst:
            assert int(te.main_1175(s)[-1][0]) == eval(s.replace("^", "**").replace("/", "//"))
        return


if __name__ == '__main__':
    unittest.main()
