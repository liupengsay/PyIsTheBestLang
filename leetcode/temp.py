
class Solution:
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:

        n = len(scores) + 1
        lst = [[ages[i], scores[i]] for i in range(n - 1)]
        lst.append([-1, 0])
        lst.sort()

        dp = [0] * n
        for i in range(1, n):
            age, score = lst[i]
            dp[i] = score
            for j in range(i):
                if lst[j][0] == age and score + dp[j] > dp[i]:
                    dp[i] = score + dp[j]
                if lst[j][0] < age and lst[j][1] <= score and score + dp[j] > dp[i]:
                    dp[i] = score + dp[j]
        return max(dp)
