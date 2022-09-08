

from itertools import combinations


def combination(n, k):
    return combinations(list(range(n)), k)


def test_combination():
    print([item for item in combination(4, 2)])
    return


if __name__ == '__main__':
    test_combination()
