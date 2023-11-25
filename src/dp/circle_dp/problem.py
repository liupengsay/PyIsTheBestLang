"""
算法：环形线性或者区间DP
功能：计算环形数组上的操作，比较简单的方式是将数组复制成两遍进行区间或者线性DP

题目：

===================================力扣===================================
918. 环形子数组的最大和（https://leetcode.cn/problems/maximum-sum-circular-subarray/）枚举可能的最大与最小区间
1388. 3n 块披萨（https://leetcode.cn/problems/pizza-with-3n-slices/）环形区间DP，类似打家劫舍
213. 打家劫舍 II（https://leetcode.cn/problems/house-robber-ii/）环形数组DP
1888. 使二进制字符串字符交替的最少反转次数（https://leetcode.cn/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/）经典循环数组的做法，复制添加数组后枚举

===================================洛谷===================================
P1880 [NOI1995] 石子合并（https://www.luogu.com.cn/problem/P1880）环形数组区间DP合并求最大值最小值
P1121 环状最大两段子段和（https://www.luogu.com.cn/problem/P1121）环形子数组和的加强版本，只选择两段
P1043 [NOIP2003 普及组] 数字游戏（https://www.luogu.com.cn/problem/P1043）环形区间DP
P1133 教主的花园（https://www.luogu.com.cn/problem/P1133）环形动态规划

参考：OI WiKi（xx）
"""
