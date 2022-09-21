

lst = []
for i in range(1, 10000):
    st = bin(i)[2:]
    if st == st[::-1]:
        lst.append(st)

def check(st, k):
    m = len(st)
    res = 0
    for i, w in enumerate(st):
        if w == "1":
            res += k**(m-1-i)
    return str(res)

dp = [[""], [str(i) for i in range(10)]]
for k in range(2, 12):
    if k%2 == 1:
        m = k//2
        lst = []
        for st in dp[-1]:
            for i in range(10):
                lst.append(st[:m] + str(i) + st[m:])
        dp.append(lst)
    else:
        lst = []
        for st in dp[-2]:
            for i in range(10):
                lst.append(str(i) + st + str(i))
        dp.append(lst)

nums = []
for lst in dp:
    for num in lst:
        if num and num[0] != "0":
            nums.append(int(num))
print(len(nums))
print(len(nums[:100]))


