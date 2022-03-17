
class Solution:
    @staticmethod
    def num_str(s):
        stack = []
        for i, va in enumerate(s):
            if va != ']':
                stack.append(va)
            elif va == ']':
                st = ''
                while stack:
                    if stack[-1] != '[':
                        st = stack.pop(-1) + st
                    else:
                        break
                stack.pop(-1)
                num = ''
                while stack:
                    if stack[-1].isnumeric():
                        num = stack.pop(-1) + num
                    else:
                        break
                if num == '':
                    num = 1
                stack.extend(list(int(num)*st))
        return ''.join(stack)

    @staticmethod
    def two_sum(nums, target):
        return True
