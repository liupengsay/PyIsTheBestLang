class SuffixArray:

    def __init__(self):
        return

    @staticmethod
    def build(s, sig):
        # sa: index is rank and value is pos
        # rk: index if pos and value is rank
        # height: lcp of rank i-th suffix and (i-1)-th suffix
        # sum(height): count of same substring of s
        # n*(n+1)//2 - sum(height): count of different substring of s
        # max(height): can compute the longest duplicate substring,
        # which is s[i: i + height[j]] and j = height.index(max(height)) and i = sa[j]
        # sig: number of unique rankings which initially is the size of the character set

        n = len(s)
        sa = list(range(n))
        rk = s[:]
        ll = 0  # ll is the length that has already been sorted, and now it needs to be sorted by 2ll length
        tmp = [0] * n
        while True:
            p = [i for i in range(n - ll, n)] + [x - ll for i, x in enumerate(sa) if x >= ll]
            # for suffixes with a length less than l, their second keyword ranking is definitely
            # the smallest because they are all empty
            # for suffixes of other lengths, suffixes starting at 'sa [i]' rank i-th, and their
            # first ll characters happen to be the second keyword of suffixes starting at 'sa[i] - ll'
            # start cardinality sorting, and first perform statistics on the first keyword
            # first, count how many values each has
            cnt = [0] * sig
            for i in range(n):
                cnt[rk[i]] += 1
            # make a prefix and for easy cardinality sorting
            for i in range(1, sig):
                cnt[i] += cnt[i - 1]

            # then use cardinality sorting to calculate the new sa
            for i in range(n - 1, -1, -1):
                w = rk[p[i]]
                cnt[w] -= 1
                sa[cnt[w]] = p[i]

            # new_sa to check new_rk
            def equal(ii, jj, lll):
                if rk[ii] != rk[jj]:
                    return False
                if ii + lll >= n and jj + lll >= n:
                    return True
                if ii + lll < n and jj + lll < n:
                    return rk[ii + lll] == rk[jj + lll]
                return False

            sig = -1
            for i in range(n):
                tmp[i] = 0

            for i in range(n):
                # compute the lcp
                if i == 0 or not equal(sa[i], sa[i - 1], ll):
                    sig += 1
                tmp[sa[i]] = sig

            for i in range(n):
                rk[i] = tmp[i]
            sig += 1
            if sig == n:
                break
            ll = ll << 1 if ll > 0 else 1

        # height
        k = 0
        height = [0] * n
        for i in range(n):
            if rk[i] > 0:
                j = sa[rk[i] - 1]
                while i + k < n and j + k < n and s[i + k] == s[j + k]:
                    k += 1
                height[rk[i]] = k
                k = 0 if k - 1 < 0 else k - 1
        return sa, rk, height
