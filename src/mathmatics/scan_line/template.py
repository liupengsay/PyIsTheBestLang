import heapq
from typing import List


class ScanLine:
    def __init__(self):
        return

    @staticmethod
    def get_sky_line(buildings: List[List[int]]) -> List[List[int]]:

        # scan_line提取建筑物轮廓
        events = []
        # 生成左右端点事件并sorting
        for left, right, height in buildings:
            # [x,y,h]分别为左右端点与高度
            events.append([left, -height, right])
            events.append([right, 0, 0])
        events.sort()

        # 初始化结果与前序高度
        res = [[0, 0]]
        stack = [[0, float('inf')]]
        for left, height, right in events:
            # 超出管辖范围的先出队
            while left >= stack[0][1]:
                heapq.heappop(stack)
            # |入备选天际线队列
            if height < 0:
                heapq.heappush(stack, [height, right])
            # 高度发生变化出现新的关键点
            if res[-1][1] != -stack[0][0]:
                res.append([left, -stack[0][0]])
        return res[1:]

    @staticmethod
    def get_rec_area(rectangles: List[List[int]]) -> int:

        # scan_line提取矩形x轴的端点并sorting（也可以取y轴的端点是一个意思）
        axis = set()
        # [x1,y1,x2,y2] 为左下角到右上角坐标
        for rec in rectangles:
            axis.add(rec[0])
            axis.add(rec[2])
        axis = sorted(list(axis))
        ans = 0
        n = len(axis)
        for i in range(n - 1):

            # brute_force两个点之间的宽度
            x1, x2 = axis[i], axis[i + 1]
            width = x2 - x1
            if not width:
                continue

            # 这里本质上是求一维区间的合并覆盖长度作为高度
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

            # 表示区[x1,x2]内的矩形覆盖高度为height
            ans += width * height
        return ans
