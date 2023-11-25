"""
算法：扩展欧几里得定理、extended_gcd、binary_gcd、二进制gcd、裴蜀定理
功能：用于求解单个同余方程
题目：

===================================力扣===================================
365. 水壶问题（https://leetcode.cn/problems/water-and-jug-problem/）经典裴蜀定理贪心
2543. 判断一个点是否可以到达（https://leetcode.cn/contest/biweekly-contest-96/problems/check-if-point-is-reachable/）利用binary_gcd的与扩展欧几里得求gcd的思想快速求解，判断可达性

===================================洛谷===================================
P1082 [NOIP2012 提高组] 同余方程（https://www.luogu.com.cn/problem/P1082）转化为同余方程求解最小的正整数解
P5435 基于值域预处理的快速 GCD（https://www.luogu.com.cn/problem/P5435）binary_gcd快速求解
P5582 【SWTR-01】Escape（https://www.luogu.com.cn/problem/P5582）贪心加脑筋急转弯，使用扩展欧几里得算法gcd为1判断可达性
P1516 青蛙的约会（https://www.luogu.com.cn/problem/P1516）求解a*x+b*y=m的最小正整数解


===================================AcWing===================================
4296. 合适数对（https://www.acwing.com/problem/content/4299/）扩展欧几里得求解ax+by=n的非负整数解

参考：OI WiKi（xx）
"""
