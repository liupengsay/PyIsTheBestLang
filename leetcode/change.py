st = """执行用时：
1160 ms
, 在所有 Python3 提交中击败了
79.29%
的用户
内存消耗：
15.1 MB
, 在所有 Python3 提交中击败了
50.71%
的用户
通过测试用例：
60 / 60"""
st = st.split("\n")
i, j = st.index('内存消耗：'), st.index('通过测试用例：')
st[i-2] = " " + st[i-2] + " "
st[j-2] = " " + st[j-2] + " "
print("- " + "".join(st[:i]))
print("- " + "".join(st[i:j]))
print("- " + "".join(st[j:]))