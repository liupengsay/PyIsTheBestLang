

# Tarjan

## 算法功能
Tarjan 算法是基于深度优先搜索的算法，用于求解图的连通性问题，参考[60 分钟搞定图论中的 Tarjan 算法]

- Tarjan 算法可以在线性时间内求出**无向图的割点与桥**，进一步地可以求解**无向图的双连通分量**
- Tarjan 算法可以也可以求解**有向图的强连通分量**，进一步地可以**求有向图的必经点与必经边**

[60 分钟搞定图论中的 Tarjan 算法]: https://zhuanlan.zhihu.com/p/101923309

## 算法伪代码

## 算法模板与测试用例
- 见Tarjan.py

## 经典题目
- 无向有环图求割点[1568. 使陆地分离的最少天数]
- 无向有环图求点最近的环[2204. Distance to a Cycle in Undirected Graph]
- 无向有环图求割边[1192. 查找集群内的「关键连接」]
- 有向有环图求环[2360. 图中的最长环]

[1192. 查找集群内的「关键连接」]: https://leetcode.cn/problems/critical-connections-in-a-network/solution/by-liupengsay-dlc2/
[2360. 图中的最长环]: https://leetcode.cn/problems/longest-cycle-in-a-graph/solution/by-liupengsay-4ff6/
[2204. Distance to a Cycle in Undirected Graph]: https://leetcode.cn/problems/distance-to-a-cycle-in-undirected-graph/solution/er-xu-cheng-ming-jiu-xu-zui-python3tarja-09qn/
[1568. 使陆地分离的最少天数]: https://leetcode.cn/problems/minimum-number-of-days-to-disconnect-island/solution/by-liupengsay-zd7w/

