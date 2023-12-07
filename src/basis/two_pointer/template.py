INF = int(1e64)


class TwoPointer:
    def __init__(self):
        return

    @staticmethod
    def window(nums):
        n = len(nums)
        ans = j = 0
        dct = dict()
        for i in range(n):
            while j < n and (nums[j] in dct or not dct
                             or (abs(max(dct) - nums[j]) <= 2 and abs(min(dct) - nums[j]) <= 2)):
                dct[nums[j]] = dct.get(nums[j], 0) + 1
                j += 1
            ans += j - i
            dct[nums[i]] -= 1
            if not dct[nums[i]]:
                del dct[nums[i]]
        return ans

    @staticmethod
    def circle_array(arr):
        """circular array pointer movement"""
        n = len(arr)
        ans = 0
        for i in range(n):
            ans = max(ans, arr[i] + arr[(i + n - 1) % n])
        return ans

    @staticmethod
    def fast_and_slow(head):
        """fast and slow pointers to determine whether there are rings in the linked list"""
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    @staticmethod
    def same_direction(nums):
        """two pointers in the same direction to find the longest subsequence without repeating elements"""
        n = len(nums)
        ans = j = 0
        pre = set()
        for i in range(n):
            while j < n and nums[j] not in pre:
                pre.add(nums[j])
                j += 1
            ans = ans if ans > j - i else j - i
            pre.discard(nums[i])
        return ans

    @staticmethod
    def opposite_direction(nums, target):
        """two pointers in the opposite direction to find two numbers equal target in ascending array"""
        n = len(nums)
        i, j = 0, n - 1
        while i < j:
            cur = nums[i] + nums[j]
            if cur > target:
                j -= 1
            elif cur < target:
                i += 1
            else:
                return True
        return False


class SlidingWindowAggregation:
    """SlidingWindowAggregation

    Api:
    1. append value to tail,O(1).
    2. pop value from head,O(1).
    3. query aggregated value in window,O(1).
    """

    def __init__(self, e, op):
        # Sliding Window Maintenance and Query Aggregated Information
        """
        Args:
            e: unit element
            op: range_merge_to_disjoint function
        """
        self.stack0 = []
        self.agg0 = []
        self.stack2 = []
        self.stack3 = []
        self.e = e
        self.e0 = self.e
        self.e1 = self.e
        self.size = 0
        self.op = op

    def append(self, value) -> None:
        if not self.stack0:
            self.push0(value)
            self.transfer()
        else:
            self.push1(value)
        self.size += 1

    def popleft(self) -> None:
        if not self.size:
            return
        if not self.stack0:
            self.transfer()
        self.stack0.pop()
        self.stack2.pop()
        self.e0 = self.stack2[-1] if self.stack2 else self.e
        self.size -= 1

    def query(self):
        return self.op(self.e0, self.e1)

    def push0(self, value):
        self.stack0.append(value)
        self.e0 = self.op(value, self.e0)
        self.stack2.append(self.e0)

    def push1(self, value):
        self.agg0.append(value)
        self.e1 = self.op(self.e1, value)
        self.stack3.append(self.e1)

    def transfer(self):
        while self.agg0:
            self.push0(self.agg0.pop())
        while self.stack3:
            self.stack3.pop()
        self.e1 = self.e

    def __len__(self):
        return self.size
