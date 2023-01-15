



dct = dict()
dct['IV'] = 4
dct['IX'] = 9
dct['XL'] = 40
dct['XC'] = 90
dct['CD'] = 400
dct['CM'] = 900
dct['I'] = 1
dct['V'] = 5
dct['X'] = 10
dct['L'] = 50
dct['C'] = 100
dct['D'] = 500
dct['M'] = 1000


print(dct)
class Solution:
    def romanToInt(self, s: str) -> int:
        dct = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900, 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        ans = i = 0
        n = len(s)
        while i < n:
            if i + 1 < n and s[i:i + 2] in dct:
                ans += dct[s[i:i + 2]]
                i += 2
            else:
                ans += dct[s[i]]
                i += 1
        return ans