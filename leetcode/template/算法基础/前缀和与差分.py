

def get_diff_array(n, shifts):
    diff = [0] * n
    for i, j, d in shifts:
        if d == 1:
            if j + 1 < n:
                diff[j + 1] -= 1
            diff[i] += 1
        else:
            if j + 1 < n:
                diff[j + 1] += 1
            diff[i] -= 1

    lst = [diff[0]]
    for d in diff[1:]:
        lst.append(lst[-1] + d)
    return lst


def test_get_diff_array():
    n = 3
    shifts = [[0, 1, 1], [1, 2, -1]]
    assert get_diff_array(n, shifts) == [1, 0, -1]
    return


if __name__ == '__main__':
    test_get_diff_array()

