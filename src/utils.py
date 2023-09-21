

def ac_max(a, b):
    return a if a > b else b


def ac_min(a, b):
    return a if a < b else b


def print_info(st):
    st = """执行结果：
    通过
    显示详情
    查看示例代码
    00 : 00 : 04

    执行用时：
    108 ms
    , 在所有 Python3 提交中击败了
    23.15%
    的用户
    内存消耗：
    15.3 MB
    , 在所有 Python3 提交中击败了
    27.31%
    的用户
    通过测试用例：
    219 / 219"""
    lst = st.split("\n")
    lst[2] = " " + lst[2] + " "
    lst[-4] = " " + lst[-4] + " "
    lst[-9] = " " + lst[-9] + " "
    lst[4] = " " + lst[4].replace(" ", "") + " "
    lst[-1] = lst[-1].replace(" ", "")
    st1 = lst[:6]
    st2 = lst[6:11]
    st3 = lst[11:-2]
    st4 = lst[-2:]
    for s in [st1, st2, st3, st4]:
        print("- " + "".join(s))
    return
