
class TestGeneral(unittest.TestCase):

    def test_extend_gcd(self):

        for _ in range(1000):
            a = random.randint(1, 10**9)
            b = random.randint(1, 10**9)
            assert ExtendGcd().binary_gcd(a, b) == math.gcd(a, b)
        return


if __name__ == '__main__':
    unittest.main()
