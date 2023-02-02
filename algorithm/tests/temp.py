


def check(s):
    num = sum(int(d) for d in str(s))
    if num >= 10:
        return check(num)
    return num

for i in range(1000, 1200):
    ans = (i-1000)%9
    assert check(i) == ans + 1