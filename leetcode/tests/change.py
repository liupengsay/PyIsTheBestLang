st = """执行用时：
328 ms
, 在所有 Python3 提交中击败了
30.28%
的用户
内存消耗：
16.9 MB
, 在所有 Python3 提交中击败了
30.89%
的用户
通过测试用例：
42 / 42"""
st = st.split("\n")
i, j = st.index('内存消耗：'), st.index('通过测试用例：')
st[i-2] = " " + st[i-2] + " "
st[j-2] = " " + st[j-2] + " "
print("- " + "".join(st[:i]))
print("- " + "".join(st[i:j]))
print("- " + "".join(st[j:]))