class SegmentTreeBitSet:
    # Use bit operations to simulate line segment trees to perform interval 01 flip operations
    def __init__(self):
        self.val = 0
        return

    def update(self, b, c):
        # The index starts from 0 and flips the interval [b, c]
        assert 0 <= b <= c
        p = (1 << (c + 1)) - (1 << b)
        self.val ^= p
        return

    def query(self, b, c):
        # The index starts from 0 to query the number 1 in intervals [b, c]
        p = (1 << (c + 1)) - (1 << b)
        return (self.val & p).bit_count()
