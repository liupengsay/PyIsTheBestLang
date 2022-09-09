

def manacher(s):
    # 马拉车算法
    n = len(s)
    arm = [0] * n
    l, r = 0, -1
    for i in range(0, n):
        k = 1 if i > r else min(arm[l + r - i], r - i + 1)

        # 持续增加回文串的长度
        while 0 <= i - k and i + k < n and s[i - k] == s[i + k]:
            k += 1
        arm[i] = k

        # 更新右侧最远的回文串边界
        k -= 1
        if i + k > r:
            l = i - k
            r = i + k
    # 返回每个位置往右的臂长
    return arm



def minCut(s: str) -> int:
    # 获取区间的回文串信息
    n = len(s)
    t = "#" + "#".join(list(s)) + "#"
    dp = manacher(t)
    m = len(t)

    lst = []
    i = 0
    for j, w in enumerate(t):
        if w != "#":
            lst.append(i)
            i += 1
        else:
            lst.append(-1)

    # 以右边界为结尾的回文子串索引
    ref = [[] for _ in range(n)]
    for j in range(m):
        left = j - dp[j] + 1
        right = j + dp[j] - 1
        while left <= right:
            if lst[left] != -1:
                ref[lst[right]].append(lst[left])
            left += 1
            right -= 1

    # 以左边界为结尾的索引
    post = [[] for _ in range(n)]
    for j in range(m):
        left = j - dp[j] + 1
        right = j + dp[j] - 1
        while left <= right:
            if lst[left] != -1:
                post[lst[left]].append(lst[right])
            left += 1
            right -= 1
    return ref, post
