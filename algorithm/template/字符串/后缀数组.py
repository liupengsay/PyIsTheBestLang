def doubling(s):
    # sa[i]:排名为i的后缀的起始位置
    # rk[i]:起始位置为i的后缀的排名
    n = len(s)
    sa = []
    rk = []
    for i in range(n):
        rk.append(ord(s[i])-ord('a')) # 刚开始时，每个后缀的排名按照它们首字母的排序
        sa.append(i) # 而排名第i的后缀就是从i开始的后缀

    l = 0 # l是已经排好序的长度，现在要按2l长度排序
    sig = 26 # sig是unique的排名的个数，初始是字符集的大小
    while True:
        p = []
        # 对于长度小于l的后缀来说，它们的第二关键字排名肯定是最小的，因为都是空的
        for i in range(n-l,n):
            p.append(i)
        # 对于其它长度的后缀来说，起始位置在`sa[i]`的后缀排名第i，而它的前l个字符恰好是起始位置为`sa[i]-l`的后缀的第二关键字
        for i in range(n):
            if sa[i]>=l:
                p.append(sa[i]-l)
        # 然后开始基数排序，先对第一关键字进行统计
        # 先统计每个值都有多少
        cnt = [0]*sig
        for i in range(n):
            cnt[rk[i]] += 1
        # 做个前缀和，方便基数排序
        for i in range(1,sig):
            cnt[i] += cnt[i-1]
        # 然后利用基数排序计算新sa
        for i in range(n-1,-1,-1):
            cnt[rk[p[i]]] -= 1
            sa[cnt[rk[p[i]]]] = p[i]
        # 然后利用新sa计算新rk
        def equal(i,j,l):
            if rk[i]!=rk[j]:return False
            if i+l>=n and j+l>=n:
                return True
            if i+l<n and j+l<n:
                return rk[i+l]==rk[j+l]
            return False
        sig = -1
        tmp = [None]*n
        for i in range(n):
            # 直接通过判断第一关键字的排名和第二关键字的排名来确定它们的前2l个字符是否相同
            if i==0 or not equal(sa[i],sa[i-1],l):
                sig += 1
            tmp[sa[i]] = sig
        rk = tmp
        sig += 1
        if sig==n:
            break
        # 更新有效长度
        l = l << 1 if l > 0 else 1
    # 计算height数组
    k = 0
    height = [0]*n
    for i in range(n):
        if rk[i]>0:
            j = sa[rk[i]-1]
            while i+k<n and j+k<n and s[i+k]==s[j+k]:
                k += 1
            height[rk[i]] = k
            k = max(0,k-1) # 下一个height的值至少从max(0,k-1)开始
    return sa,rk,height



def test_get_doubling():
    word1 = "abcabc"
    word2 = "abdcaba"
    word = word1 + "a" + word2
    sa, rk, height = doubling(word)
    m, n = len(word1), len(word2)
    i = 0
    j = 0
    merge = ""
    while i<m and j<n:
        if rk[i] > rk[j+m+1]:
            merge += word1[i]
            i += 1
        else:
            merge += word2[j]
            j += 1
    merge += word1[i:]
    merge += word2[j:]
    assert merge == "abdcabcabcaba"
    return


if __name__ == '__main__':
    test_get_doubling()



print(doubling("abcde"))