from typing import List


class BitOperation:
    def __init__(self):
        return

    @staticmethod
    def sum_xor(n):
        """xor num of range(0, x+1)"""
        if n % 4 == 0:
            return n
        # (4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0
        elif n % 4 == 1:
            return 1  # n^(n-1)
        elif n % 4 == 2:
            return n + 1  # n^(n-1)^(n-2)
        return 0  # n^(n-1)^(n-2)^(n-3)

    @staticmethod
    def sum_xor_2(n):
        """xor num of range(0, x+1)"""
        if n % 4 == 0:
            return n
        # (4*i)^(4*i+1)^(4*i+2)^(4*i+3)=0
        elif n % 4 == 1:
            return n ^ (n - 1)
        elif n % 4 == 2:
            return n ^ (n - 1) ^ (n - 2)
        return n ^ (n - 1) ^ (n - 2) ^ (n - 3)

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
