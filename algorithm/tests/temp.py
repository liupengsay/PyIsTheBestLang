

n = 1000

k = 5
dct= [ [] for _ in range(n+1)]

for i in range(1, n+1):
    for j in range(i+1, n+1):
        if bin(i^j).count("1") == k:
            dct[i].append(j)
            dct[j].append(i)
print(dct)
