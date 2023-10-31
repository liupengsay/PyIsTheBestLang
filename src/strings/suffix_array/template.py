class SuffixArray:
    def __init__(self, ind: dict):
        # key if character and value is rank
        self.ind = ind
        return

    def get_array(self, s):
        # sa: index is rank and value is pos
        # rk: index if pos and value is rank
        # height: lcp of rank i-th suffix and (i-1)-th suffix
        # sum(height) is count of same substring of s
        # n*(n+1)//2 - sum(height) if count of different substring of s

        n = len(s)
        sa = []
        rk = []
        for i in range(n):
            rk.append(self.ind[s[i]])
            sa.append(i)

        ll = 0  # ll is the length that has already been sorted, and now it needs to be sorted by 2ll length
        sig = len(self.ind)  # number of unique rankings, initially the size of the character set
        while True:
            p = []
            # for suffixes with a length less than l, their second keyword ranking is definitely
            # the smallest because they are all empty
            for i in range(n - ll, n):
                p.append(i)
            # for suffixes of other lengths, suffixes starting at 'sa [i]' rank i-th, and their
            # first ll characters happen to be the second keyword of suffixes starting at 'sa[i] - ll'

            for i in range(n):
                if sa[i] >= ll:
                    p.append(sa[i] - ll)
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
                cnt[rk[p[i]]] -= 1
                sa[cnt[rk[p[i]]]] = p[i]

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
            tmp = [None] * n
            for i in range(n):
                # compute the lcp
                if i == 0 or not equal(sa[i], sa[i - 1], ll):
                    sig += 1
                tmp[sa[i]] = sig
            rk = tmp
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
                k = max(0, k - 1)  # the k value of next height
        return sa, rk, height
