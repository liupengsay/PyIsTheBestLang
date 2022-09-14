import heapq

stack = []
s = input().split()
while True:
    s = input().split()
    if len(s) == 2:
        heapq.heappush(stack, int(s[1]))
    elif s[0] == "2":
        print(stack[0])
    else:
        heapq.heappop(stack)




