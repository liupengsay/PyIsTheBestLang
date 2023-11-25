class CircleSection:
    def __init__(self):
        return

    @staticmethod
    def compute_circle_result(n: int, m: int, x: int, tm: int) -> int:
        """use hash table and list to record the first pos of circle section"""
        dct = dict()
        # example is x = (x + m) % n
        lst = []
        while x not in dct:
            dct[x] = len(lst)
            lst.append(x)
            x = (x + m) % n

        length = len(lst)
        # the first pos of circle section
        ind = dct[x]
        # current lst is enough
        if tm < length:
            return lst[tm]

        # compute by circle section
        circle = length - ind
        tm -= length
        j = tm % circle
        return lst[ind + j]

    @staticmethod
    def circle_section_pre(n, grid, c, sta, cur, h):
        """circle section with prefix sum"""
        dct = dict()
        lst = []
        cnt = []
        while sta not in dct:
            dct[sta] = len(dct)
            lst.append(sta)
            cnt.append(c)
            sta = cur
            c = 0
            cur = 0
            for i in range(n):
                num = 1 if sta & (1 << i) else 2
                for j in range(n):
                    if grid[i][j] == "1":
                        c += num
                        cur ^= (num % 2) * (1 << j)

        length = len(lst)
        ind = dct[sta]
        pre = [0] * (length + 1)
        for i in range(length):
            pre[i + 1] = pre[i] + cnt[i]

        ans = 0
        if h < length:
            return ans + pre[h]

        circle = length - ind
        circle_cnt = pre[length] - pre[ind]

        h -= length
        ans += pre[length]

        ans += (h // circle) * circle_cnt

        j = h % circle
        ans += pre[ind + j] - pre[ind]
        return ans
