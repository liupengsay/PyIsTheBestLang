import math
import random
from typing import List

from src.data_structure.sorted_list.template import SortedList
from src.utils.fast_io import inf


class Geometry:
    def __init__(self):
        return

    @staticmethod
    def compute_center(x1, y1, x2, y2, r):
        # Calculate the centers of two circles passing through two different points and determining the radius
        px, py = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x1 - x2, y1 - y2
        h = math.sqrt(r * r - (dx * dx + dy * dy) / 4)
        res = []
        for fx, fy in ((1, -1), (-1, 1)):
            cx = px + fx * h * dy / math.sqrt(dx * dx + dy * dy)
            cy = py + fy * h * dx / math.sqrt(dx * dx + dy * dy)
            res.append([cx, cy])
        return res

    @staticmethod
    def same_line(point1, point2, point3):
        # calculating three point collinearity
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        return (x2 - x1) * (y3 - y2) == (x3 - x2) * (y2 - y1)

    @staticmethod
    def compute_slope2(point1, point2):
        assert point1 != point2
        # Determine the slope of a straight line based on two different points
        x1, y1 = point1
        x2, y2 = point2
        a, b = x2 - x1, y2 - y1
        g = math.gcd(a, b)
        a //= g
        b //= g
        if a < 0:
            a *= -1
            b *= -1
        elif a == 0:
            b = abs(b)
        return a, b

    @staticmethod
    def compute_slope(x1, y1, x2, y2):
        assert [x1, y1] != [x2, y2]
        # Determine the slope of a straight line based on two different points
        if x1 == x2:
            ans = "x"
        else:
            a = y2 - y1
            b = x2 - x1
            g = math.gcd(a, b)
            if b < 0:
                a *= -1
                b *= -1
            ans = [a // g, b // g]
        return ans

    @staticmethod
    def compute_square_point(x0, y0, x2, y2):
        assert [x0, y0] != [x2, y2]
        # Given two points on the diagonal of a rectangle and ensuring that they are different
        # calculate the coordinates of the other two points
        x1 = (x0 + x2 + y2 - y0) / 2
        y1 = (y0 + y2 + x0 - x2) / 2
        x3 = (x0 + x2 - y2 + y0) / 2
        y3 = (y0 + y2 - x0 + x2) / 2
        return (x1, y1), (x3, y3)

    @staticmethod
    def compute_square_point_2(x0, y0, x2, y2):
        # Given two points on the diagonal of a rectangle and ensuring that they are different
        # calculate the coordinates of the other two points
        assert [x0, y0] != [x2, y2]
        # Judging a square
        assert abs(x0 - x2) == abs(y0 - y2)
        return (x0, y2), (x2, y0)

    @staticmethod
    def compute_square_area(x0, y0, x2, y2):
        # Given the points on the diagonal of a square
        # calculate the area of the square, taking into account that it is an integer

        ans = (x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)
        return ans // 2

    @staticmethod
    def compute_triangle_area(x1, y1, x2, y2, x3, y3):
        # Can be used to determine the positional relationship between points and triangles
        return abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3)) / 2

    @staticmethod
    def line_intersection_line(start1: List[int], end1: List[int], start2: List[int], end2: List[int]) -> List[float]:
        # Calculate the intersection point of two line segments that are bottommost and leftmost
        # If there is no intersection point, return empty
        x1, y1 = start1
        x2, y2 = end1
        x3, y3 = start2
        x4, y4 = end2
        det = lambda a, b, c, dd: a * dd - b * c
        d = det(x1 - x2, x4 - x3, y1 - y2, y4 - y3)
        p = det(x4 - x2, x4 - x3, y4 - y2, y4 - y3)
        q = det(x1 - x2, x4 - x2, y1 - y2, y4 - y2)
        if d != 0:
            lam, eta = p / d, q / d
            if not (0 <= lam <= 1 and 0 <= eta <= 1):
                return []
            return [lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2]
        if p != 0 or q != 0:
            return []
        t1, t2 = sorted([start1, end1]), sorted([start2, end2])
        if t1[1] < t2[0] or t2[1] < t1[0]:
            return []
        return max(t1[0], t2[0])


class ClosetPair:
    def __init__(self):
        return

    @staticmethod
    def bucket_grid(n: int, nums: List[List[int]]):
        # Use random increment method to divide the grid and calculate the closest point pairs on the plane
        def dis(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        def check(p):
            nonlocal ss
            return p[0] // ss, p[1] // ss

        def update(buck, ind):
            nonlocal dct
            if buck not in dct:
                dct[buck] = []
            dct[buck].append(ind)
            return

        assert n >= 2
        random.shuffle(nums)

        # initialization
        dct = dict()
        ans = dis(nums[0], nums[1])
        ss = ans ** 0.5
        update(check(nums[0]), 0)
        update(check(nums[1]), 1)
        if ans == 0:
            return 0

        # traverse with random increments
        for i in range(2, n):
            a, b = check(nums[i])
            res = ans
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    cur = (x + a, y + b)
                    if cur in dct:
                        for j in dct[cur]:
                            now = dis(nums[i], nums[j])
                            res = res if res < now else now
            if res == 0:  # Directly return at a distance of 0
                return 0
            if res < ans:
                # initialization again
                ans = res
                ss = ans ** 0.5
                dct = dict()
                for x in range(i + 1):
                    update(check(nums[x]), x)
            else:
                update(check(nums[i]), i)
        # The return value is the square of the Euclidean distance
        return ans

    @staticmethod
    def divide_and_conquer(lst):

        # Using Divide and Conquer to Solve the Pairs of Nearest Points in a Plane
        lst.sort(key=lambda p: p[0])

        def distance(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        def conquer(point_set, min_dis, mid):
            smallest = min_dis
            point_mid = point_set[mid]
            pt = []
            for point in point_set:
                if (point[0] - point_mid[0]) ** 2 <= min_dis:
                    pt.append(point)
            pt.sort(key=lambda x: x[1])
            for i in range(len(pt)):
                for j in range(i + 1, len(pt)):
                    if (pt[i][1] - pt[j][1]) ** 2 >= min_dis:
                        break
                    cur = distance(pt[i], pt[j])
                    smallest = smallest if smallest < cur else cur
            return smallest

        def check(point_set):
            min_dis = inf
            if len(point_set) <= 3:
                n = len(point_set)
                for i in range(n):
                    for j in range(i + 1, n):
                        cur = distance(point_set[i], point_set[j])
                        min_dis = min_dis if min_dis < cur else cur
                return min_dis
            mid = len(point_set) // 2
            left = point_set[:mid]
            right = point_set[mid + 1:]
            min_left_dis = check(left)
            min_right_dis = check(right)
            min_dis = min_left_dis if min_left_dis < min_right_dis else min_right_dis
            range_merge_to_disjoint_dis = conquer(point_set, min_dis, mid)
            return min_dis if min_dis < range_merge_to_disjoint_dis else range_merge_to_disjoint_dis

        return check(lst)

    @staticmethod
    def sorted_pair(points) -> float:
        # Using an ordered list for calculating the closest point pairs on a plane
        def dis(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        points.sort(key=lambda p: [p[0], [1]])
        lst1 = SortedList()
        lst2 = SortedList()
        ans = inf
        ss = ans ** 0.5
        n = len(points)
        for i in range(n):
            x, y = points[i]
            while lst1 and abs(x - lst1[0][0]) >= ss:
                a, b = lst1.pop(0)
                lst2.discard((b, a))
            while lst1 and abs(x - lst1[-1][0]) >= ss:
                a, b = lst1.pop()
                lst2.discard((b, a))

            ind = lst2.bisect_left((y - ss, -inf))
            while ind < len(lst2) and abs(y - lst2[ind][0]) <= ss:
                res = dis([x, y], lst2[ind][::-1])
                ans = ans if ans < res else res
                ind += 1

            ss = ans ** 0.5
            lst1.add((x, y))
            lst2.add((y, x))
        return ans

    @staticmethod
    def bucket_grid_inter_set(n: int, nums1: List[List[int]], nums2):

        # Using the Random Incremental Method to Divide Grids and
        # Calculate the Nearest Point Pairs of Two Planar Point Sets

        def dis(p1, p2):
            return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

        def check(p):
            nonlocal ss
            return p[0] // ss, p[1] // ss

        def update(buck, ind):
            nonlocal dct
            if buck not in dct:
                dct[buck] = []
            dct[buck].append(ind)
            return

        random.shuffle(nums1)
        random.shuffle(nums2)

        dct = dict()
        ans = inf
        for i in range(n):
            cur = dis(nums1[i], nums2[0])
            ans = cur if ans > cur else ans
        ss = ans ** 0.5
        if ans == 0:
            return 0
        for i in range(n):
            update(check(nums1[i]), i)

        for i in range(1, n):
            a, b = check(nums2[i])
            res = ans
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    cur = (x + a, y + b)
                    if cur in dct:
                        for j in dct[cur]:
                            now = dis(nums2[i], nums1[j])
                            res = res if res < now else now
            if res == 0:
                return 0
            if res < ans:
                ans = res
                ss = ans ** 0.5
                dct = dict()
                for x in range(n):
                    update(check(nums1[x]), x)
        # The return value is the square of the Euclidean distance
        return ans
