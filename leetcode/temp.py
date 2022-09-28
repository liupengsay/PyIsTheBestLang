
dct = {"0":"0", "1":"1", "6":"9", "9":"6", "8": "8"}

def check(st):
    if st and st[0] == "0":
        return -1
    if not 1<=len(st)<=9:
        return -1
    ori = int(st)
    rev = "".join([dct[x] for x in st[::-1]])
    if int(rev) != ori:
        return ori
    return -1

def dfs(pre):
    if len(pre) == 10:
        return
    num = check(pre)
    if num != -1:
        dp.add(num)
    for s in "01689":
        dfs(pre+s)
    return

dp = set()
dfs("")
dp = sorted(list(dp))

print(dp[:100])