from typing import List

from src.data_structure.sorted_list.template import SortedList


class MinimumPairXor:
    def __init__(self):
        """
        if x < y < z then min(x^y, y^z) < x^z, thus the minimum xor pair must be adjacent
        """
        self.lst = SortedList()
        self.xor = SortedList()
        return

    def add(self, num):
        i = self.lst.bisect_left(num)
        if i < len(self.lst):
            if 0 <= i - 1:
                self.xor.discard(self.lst[i] ^ self.lst[i - 1])
        self.lst.add(num)
        if 0 <= i - 1 < len(self.lst):
            self.xor.add(num ^ self.lst[i - 1])
        if 0 <= i + 1 < len(self.lst):
            self.xor.add(num ^ self.lst[i + 1])
        return

    def remove(self, num):
        i = self.lst.bisect_left(num)
        if 0 <= i - 1 < len(self.lst):
            self.xor.discard(num ^ self.lst[i - 1])
        if 0 <= i + 1 < len(self.lst):
            self.xor.discard(num ^ self.lst[i + 1])
        self.lst.discard(num)
        if i < len(self.lst) and i - 1 >= 0:
            self.xor.add(self.lst[i] ^ self.lst[i - 1])
        return

    def query(self):
        return self.xor[0]


class BitOperation:
    def __init__(self):
        return

    @staticmethod
    def sum_xor(n):
        """xor num of range(0, x+1)"""
        if n % 4 == 0:
            return n  # (4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0
        elif n % 4 == 1:
            return 1  # n^(n-1)
        elif n % 4 == 2:
            return n + 1  # n^(n-1)^(n-2)
        return 0  # n^(n-1)^(n-2)^(n-3)

    @staticmethod
    def graycode_to_integer(graycode):
        graycode_len = len(graycode)
        binary = list()
        binary.append(graycode[0])
        for i in range(1, graycode_len):
            if graycode[i] == binary[i - 1]:
                b = 0
            else:
                b = 1
            binary.append(str(b))
        return int("0b" + ''.join(binary), 2)

    @staticmethod
    def integer_to_graycode(integer):
        binary = bin(integer).replace('0b', '')
        graycode = list()
        binary_len = len(binary)
        graycode.append(binary[0])
        for i in range(1, binary_len):
            if binary[i - 1] == binary[i]:
                g = 0
            else:
                g = 1
            graycode.append(str(g))
        return ''.join(graycode)

    @staticmethod
    def get_graycode(n: int) -> List[int]:
        """all graycode number whose length small or equal to n"""
        code = [0, 1]
        for i in range(1, n):
            code.extend([(1 << i) + num for num in code[::-1]])
        return code
