class ReadFile:
    def __init__(self, path):
        self.fr = open(path, "r", encoding="utf-8", errors="ignore")

    def close(self):
        self.fr.close()

    def read_int(self):
        return int(self.fr.readline().rstrip())

    def read_float(self):
        return float(self.fr.readline().rstrip())

    def read_list_ints(self):
        return list(map(int, self.fr.readline().rstrip().split()))

    def read_list_floats(self):
        return list(map(float, self.fr.readline().rstrip().split()))

    def read_list_ints_minus_one(self):
        return list(map(lambda x: int(x) - 1, self.fr.readline().rstrip().split()))

    def read_str(self):
        return self.fr.readline().rstrip()

    def read_list_strs(self):
        return self.fr.readline().rstrip().split()

    def read_list_str(self):
        return list(self.fr.readline().rstrip())

    @staticmethod
    def st(s):
        print(s)
        return
