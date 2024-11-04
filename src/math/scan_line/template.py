import heapq
from typing import List


class ScanLine:
    def __init__(self):
        return

    @staticmethod
    def get_sky_line(buildings: List[List[int]]) -> List[List[int]]:

        events = []
        for left, right, height in buildings:
            events.append([left, -height, right])
            events.append([right, 0, 0])
        events.sort()

        res = [[0, 0]]
        stack = [[0, float('math.inf')]]
        for left, height, right in events:
            while left >= stack[0][1]:
                heapq.heappop(stack)
            if height < 0:
                heapq.heappush(stack, [height, right])
            if res[-1][1] != -stack[0][0]:
                res.append([left, -stack[0][0]])
        return res[1:]

    @staticmethod
    def get_rec_area(rectangles: List[List[int]]) -> int:

        axis = set()
        # [x1,y1,x2,y2] left_down to right_up
        for rec in rectangles:
            axis.add(rec[0])
            axis.add(rec[2])
        axis = sorted(list(axis))
        ans = 0
        n = len(axis)
        for i in range(n - 1):

            x1, x2 = axis[i], axis[i + 1]
            width = x2 - x1
            if not width:
                continue

            items = [[rec[1], rec[3]] for rec in rectangles if rec[0] < x2 and rec[2] > x1]
            items.sort(key=lambda x: [x[0], -x[1]])
            height = low = high = 0
            for y1, y2 in items:
                if y1 >= high:
                    height += high - low
                    low, high = y1, y2
                else:
                    high = high if high > y2 else y2
            height += high - low

            ans += width * height
        return ans
