class SegmentTreeBitSet:
    # 使用位运算模拟线段树进行区间01翻转操作
    def __init__(self):
        self.val = 0
        return

    def update(self, b, c):
        # 索引从0开始翻转区间[b, c]
        p = (1 << (c + 1)) - (1 << b)
        self.val ^= p
        return

    def query(self, b, c):
        # 索引从0开始查询区间[b, c]的个数
        p = (1 << (c + 1)) - (1 << b)
        return (self.val & p).bit_count()
