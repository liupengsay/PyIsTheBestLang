class SegBitSet:
    def __init__(self, n):
        self.n = n
        self.val = 0
        return

    def update(self, ll, rr):
        assert 0 <= ll <= rr <= self.n - 1
        mask = ((1 << (rr - ll + 1)) - 1) << ll
        self.val ^= mask
        return

    def query(self, ll, rr):
        assert 0 <= ll <= rr <= self.n - 1
        if ll == 0 and rr == self.n - 1:
            return self.val.bit_count()
        mask = ((1 << (rr - ll + 1)) - 1) << ll
        return (self.val & mask).bit_count()
