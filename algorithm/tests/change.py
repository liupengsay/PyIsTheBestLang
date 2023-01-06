st = """执行用时：
44 ms
, 在所有 Python3 提交中击败了
22.73%
的用户
内存消耗：
15.2 MB
, 在所有 Python3 提交中击败了
9.09%
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