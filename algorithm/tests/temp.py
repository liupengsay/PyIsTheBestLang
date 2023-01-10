

import random


def check(x, n):
    if x == 1.0:
        return "1.0000"
    n = int(n)
    error = 1e-18
    ans = x
    pre = x
    for i in range(2, n + 1):
        pre *= x
        if pre / i < error:
            break
        ans += pre / i
    return "%.4f" % ans


for _ in range(100):
    n = 100
    x = random.uniform(0.0001, 1)
    ans1 = check(x, n)

    s = sum(x**i/i for i in range(1, n+1))
    ans2 = "%.4f" % s
    print(ans1, ans2)
    assert ans1 == ans2