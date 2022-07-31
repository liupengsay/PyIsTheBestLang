class Solution:
    def canTransform(self, start: str, end: str) -> bool:
        if start.replace('X', '') != end.replace('X', ''):
            return False
        n = len(start)
        start_l = [i for i in range(n) if start[i]=='L']
        start_r = [i for i in range(n) if start[i]=='R']

        end_l = [i for i in range(n) if end[i]=='L']
        end_r = [i for i in range(n) if end[i]=='R']
        if len(start_l) != len(end_l):
            return False
        if len(start_r) != len(end_r):
            return False
        for i, index in enumerate(start_l):
            if index > end_l[i]:
                return False

        for i, index in enumerate(start_r):
            if index < end_r[i]:
                return False
        return True
