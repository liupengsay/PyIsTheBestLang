

import sys


class FastIO:
    def __init__(self):
        return

    @staticmethod
    def read_str():
        return sys.stdin.readline().rstrip()

    @staticmethod
    def read_int():
        return int(sys.stdin.readline().rstrip())

    @staticmethod
    def read_list_ints():
        return list(map(int, sys.stdin.readline().rstrip().split()))

    @staticmethod
    def st(x):
        return print(x)

    @staticmethod
    def lst(x):
        return print(*x)


class Solution:
    def __init__(self):
        return

    @staticmethod
    def main(ac=FastIO()):
        n = ac.read_int()
        s = list(ac.read_str())
        zero, one, two = s.count("0"), s.count("1"), s.count("2")
        m = n // 3
        if zero < m:
            for i in range(n):
                if zero == m:
                    break
                if s[i] == "1" and one > m:
                    s[i] = "0"
                    one -= 1
                    zero += 1
                elif s[i] == "2" and two > m:
                    s[i] = "0"
                    two -= 1
                    zero += 1
        elif zero > m:
            lst = [i for i in range(n) if s[i] == "0"][m:]
            for i in lst:
                if one < m:
                    s[i] = "1"
                    one += 1
                else:
                    s[i] = "2"
                    two += 1
        if one < m:
            for i in range(n):
                if s[i] == "2":
                    s[i] = "1"
                    one += 1
                    if one == m:
                        break
        elif one > m:
            for i in range(n - 1, -1, -1):
                if two == m:
                    break
                if s[i] == "1":
                    s[i] = "2"
                    two += 1
        ac.st("".join(s))
        return


Solution().main()
