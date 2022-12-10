st = """执行用时：
352 ms
, 在所有 Python3 提交中击败了
100.00%
的用户
内存消耗：
15.4 MB
, 在所有 Python3 提交中击败了
55.56%
的用户
通过测试用例：
100 / 100"""
st = st.split("\n")
i, j = st.index('内存消耗：'), st.index('通过测试用例：')
st[i-2] = " " + st[i-2] + " "
st[j-2] = " " + st[j-2] + " "
print("- " + "".join(st[:i]))
print("- " + "".join(st[i:j]))
print("- " + "".join(st[j:]))