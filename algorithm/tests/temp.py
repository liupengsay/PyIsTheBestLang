import random
import math


for i in range(4, 10):
    ans = sum(math.comb(i, j) for j in range(0, i+1, 2))
    assert ans == 2**(i-1)