import heapq
import random
import unittest

from sortedcontainers import SortedList

"""
算法：堆（优先队列）
功能：通常用于需要贪心的场景
题目：

===================================力扣===================================
630. 课程表（https://leetcode.cn/problems/course-schedule-iii/）用一个堆延迟选择贪心维护最优
2454. 下一个更大元素 IV（https://leetcode.cn/problems/next-greater-element-iv/）使用两个堆维护下下个更大元素即出队两次时遇见的元素
2402. 会议室 III（https://leetcode.cn/problems/meeting-rooms-iii/）使用两个堆模拟进行会议室安排并进行计数
2386. 找出数组的第 K 大和（https://leetcode.cn/problems/find-the-k-sum-of-an-array/）转换思路使用堆维护最大和第 K 次出队的则为目标结果
2163. 删除元素后和的最小差值（https://leetcode.cn/problems/minimum-difference-in-sums-after-removal-of-elements/）预处理前缀后缀最大最小的 K 个数和再进行枚举分割点
1792. 最大平均通过率（https://leetcode.cn/problems/maximum-average-pass-ratio/）贪心依次给增幅最大的班级人数加 1 

===================================洛谷===================================
P1168 中位数（https://www.luogu.com.cn/problem/P1168） 用两个堆维护中位数
P1801 黑匣子（https://www.luogu.com.cn/problem/P1801）用两个堆维护第K小
P2085 最小函数值（https://www.luogu.com.cn/problem/P2085）用数学加一个堆维护前K小
P1631 序列合并（https://www.luogu.com.cn/problem/P1631）用一个堆维护前K小
P4053 建筑抢修（https://www.luogu.com.cn/problem/P4053）用一个堆延迟选择贪心维护最优
P1878 舞蹈课（https://www.luogu.com.cn/problem/P1878）用哈希加一个堆进行模拟计算

参考：OI WiKi（xx）
"""


class Solution:
    def __int__(self):
        return

    @staticmethod
    def lc_1792(classes, extra_students):
        stack = []
        for p, t in classes:
            heapq.heappush(stack, [p / t - (p + 1) / (t + 1), p, t])
        for _ in range(extra_students):
            r, p, t = heapq.heappop(stack)
            p += 1
            t += 1
            # 关键点在于优先级的设置为 p / t - (p + 1) / (t + 1)
            heapq.heappush(stack, [p / t - (p + 1) / (t + 1), p, t])
        return sum(p / t for _, p, t in stack) / len(classes)

class HeapqMedian:
    def __init__(self, mid):
        # 使用两个堆动态维护奇数长度数组的中位数
        self.mid = mid
        self.left = []
        self.right = []
        return

    def add(self, num):
        # 根据大小先放入左右两边的堆
        if num > self.mid:
            heapq.heappush(self.right, num)
        else:
            heapq.heappush(self.left, -num)
        n = len(self.left) + len(self.right)

        if n % 2 == 0:
            # 如果是奇数长度则更新中位数并保持左右两边数组长度相等
            if len(self.left) > len(self.right):
                heapq.heappush(self.right, self.mid)
                self.mid = -heapq.heappop(self.left)
            elif len(self.right) > len(self.left):
                heapq.heappush(self.left, -self.mid)
                self.mid = heapq.heappop(self.right)
        return

    def query(self):
        return self.mid


class TestGeneral(unittest.TestCase):

    def test_heapq_median(self):
        ceil = 1000
        num = random.randint(0, ceil)
        lst = SortedList([num])
        hm = HeapqMedian(num)
        for i in range(ceil):
            num = random.randint(0, ceil)
            lst.add(num)
            hm.add(num)
            if i % 2:
                assert lst[(i + 2) // 2] == hm.query()
        return


if __name__ == '__main__':
    unittest.main()
