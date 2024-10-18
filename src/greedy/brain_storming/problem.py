"""

Algorithm：greedy|reverse_thinking|pigeonhole_principle|inclusion_exclusion|custom_sort|brain_teaser|construction
Description：brain_teaser

====================================LeetCode====================================
30（https://leetcode.cn/problems/p0NxJO）greedy|regret_heapq|brain_teaser
134（https://leetcode.cn/problems/gas-station/）greedy
330（https://leetcode.cn/problems/patching-array/）greedy
1199（https://leetcode.cn/problems/minimum-time-to-build-blocks/）huffman_tree|greedy|classical|heapq
2499（https://leetcode.cn/problems/minimum-total-cost-to-make-arrays-unequal/）pigeonhole_principle|greedy
2449（https://leetcode.cn/problems/minimum-total-cost-to-make-arrays-unequal/）sort|odd_even|greedy
2448（https://leetcode.cn/problems/minimum-cost-to-make-array-equal/）median|greedy
2412（https://leetcode.cn/problems/minimum-money-required-before-transactions/）custom_sort
2366（https://leetcode.cn/problems/minimum-replacements-to-sort-the-array/）reverse_order|greedy
2350（https://leetcode.cn/problems/shortest-impossible-sequence-of-rolls/）brain_teaser|classical
2344（https://leetcode.cn/problems/minimum-deletions-to-make-array-divisible/）greedy|range_gcd
2136（https://leetcode.cn/problems/earliest-possible-day-of-full-bloom/）greedy
2071（https://leetcode.cn/problems/maximum-number-of-tasks-you-can-assign/）greedy|binary_search
517（https://leetcode.cn/problems/super-washing-machines/）greedy|binary_search|brute_force
1798（https://leetcode.cn/problems/maximum-number-of-consecutive-values-you-can-make/）greedy
625（https://leetcode.cn/problems/minimum-factorization/）greedy|factorization
2568（https://leetcode.cn/problems/minimum-impossible-or/）brain_teaser|greedy|guess|watch_pattern
6361（https://leetcode.cn/problems/minimum-score-by-changing-two-elements/）brain_teaser|greedy
6316（https://leetcode.cn/problems/rearrange-array-to-maximize-prefix-score/）greedy|prefix_sum
2436（https://leetcode.cn/problems/minimum-split-into-subarrays-with-gcd-greater-than-one/）greedy
1029（https://leetcode.cn/problems/two-city-scheduling/）greedy|sort
1353（https://leetcode.cn/problems/maximum-number-of-events-that-can-be-attended/）brute_force|greedy
1402（https://leetcode.cn/problems/reducing-dishes/）prefix_sum|greedy
1665（https://leetcode.cn/problems/minimum-initial-energy-to-finish-tasks/）greedy|sort|implemention|CF1203F
1675（https://leetcode.cn/problems/minimize-deviation-in-array/）brain_teaser|greedy
1686（https://leetcode.cn/problems/stone-game-vi/）greedy|custom_sort
1808（https://leetcode.cn/problems/maximize-number-of-nice-divisors/）mod|greedy|classical|maximum_mul
1953（https://leetcode.cn/problems/maximum-number-of-weeks-for-which-you-can-work/）greedy|classical|maximum|sum
2856（https://leetcode.cn/problems/minimum-array-length-after-pair-removals/）greedy|classical|maximum|sum
858（https://leetcode.cn/problems/mirror-reflection/description/）brain_teaser
1927（https://leetcode.cn/problems/sum-game/description/）game_dp|brain_teaser|classification_discussion
2592（https://leetcode.cn/problems/maximize-greatness-of-an-array/）classical|greedy|sort|two_pointers
1503（https://leetcode.cn/problems/last-moment-before-all-ants-fall-out-of-a-plank/）brain_teaser|classical
991（https://leetcode.cn/problems/broken-calculator/）reverse_order|reverse_thinking|greedy|odd_even|implemention
2745（https://leetcode.cn/problems/construct-the-longest-new-string/）brain_teaser|greedy
1657（https://leetcode.cn/problems/determine-if-two-strings-are-close/description/）brain_teaser|greedy
2561（https://leetcode.cn/problems/rearranging-fruits/）brain_teaser|greedy
843（https://leetcode.cn/problems/guess-the-word/）brain_teaser|greedy|implemention|interactive
1946（https://leetcode.cn/problems/largest-number-after-mutating-substring/description/）greedy|classical
1540（https://leetcode.cn/problems/can-convert-string-in-k-moves/）greedy|brain_teaser|pointer
1121（https://leetcode.cn/problems/divide-array-into-increasing-sequences/description/）brain_teaser|greedy|classical|maximum
3012（https://leetcode.com/problems/minimize-length-of-array-using-operations/）brain_teaser|perishu_theorem|hard|greedy
100197（https://leetcode.cn/problems/earliest-second-to-mark-indices-ii/description/）regret_heapq|binary_search|brain_teaser|classical
100227（https://leetcode.cn/problems/minimum-moves-to-pick-k-ones/description/）median_greedy|brute_force|implemention
100367（https://leetcode.cn/problems/minimum-cost-for-cutting-cake-ii/description/）sort|greedy|big_to_small
2576（https://leetcode.cn/problems/find-the-maximum-number-of-marked-indices）observation|greedy|classical

=====================================LuoGu======================================
P1031（https://www.luogu.com.cn/problem/P1031）greedy|prefix_sum|counter
P1286（https://www.luogu.com.cn/problem/P1286）brain_teaser|sorted_list
P1684（https://www.luogu.com.cn/problem/P1684）greedy
P1658（https://www.luogu.com.cn/problem/P1658）greedy
P2001（https://www.luogu.com.cn/problem/P2001）greedy|classical
P1620（https://www.luogu.com.cn/problem/P1620）classification_discussion|greedy
P2773（https://www.luogu.com.cn/problem/P2773）classification_discussion|greedy
P2255（https://www.luogu.com.cn/problem/P2255）pointers|greedy
P2327（https://www.luogu.com.cn/problem/P2327）brain_teaser|brute_force
P2777（https://www.luogu.com.cn/problem/P2777）greedy|brute_force|prefix_suffix
P2649（https://www.luogu.com.cn/problem/P2649）greedy|brain_teaser
P1367（https://www.luogu.com.cn/problem/P1367）brain_teaser|sort|classical
P1362（https://www.luogu.com.cn/problem/P1362）bfs|brute_force|observe_pattern
P1090（https://www.luogu.com.cn/problem/P1090）greedy|small_to_big
P1334（https://www.luogu.com.cn/problem/P1334）reverse_thinking|greedy|small_to_big
P1250（https://www.luogu.com.cn/problem/P1250）greed|segment_tree|range_update|range_sum|binary_search
P1230（https://www.luogu.com.cn/problem/P1230）sort|greedy
P1159（https://www.luogu.com.cn/problem/P1159）greedy|implemention|deque
P1095（https://www.luogu.com.cn/problem/P1095）greedy|implemention
P1056（https://www.luogu.com.cn/problem/P1056）counter|sort|greedy
P8847（https://www.luogu.com.cn/problem/P8847）classification_discussion|greedy
P8845（https://www.luogu.com.cn/problem/P8845）brain_teaser|prime_property|only_even_prime
P2772（https://www.luogu.com.cn/problem/P2772）sort|partial_order
P2878（https://www.luogu.com.cn/problem/P2878）greedy|custom_sort|classical
P2920（https://www.luogu.com.cn/problem/P2920）sort|greedy
P2983（https://www.luogu.com.cn/problem/P2983）greedy|sort
P3173（https://www.luogu.com.cn/problem/P3173）sort|greedy|big_to_small
P5098（https://www.luogu.com.cn/problem/P5098）greedy|partial_order|classification_discussion|prefix_min|sort
P5159（https://www.luogu.com.cn/problem/P5159）xor_property|brute_force|counter|fast_power
P5497（https://www.luogu.com.cn/problem/P5497）pigeonhole_principle|classification_discussion
P5682（https://www.luogu.com.cn/problem/P5682）brain_teaser|sort|greedy|brute_force
P5804（https://www.luogu.com.cn/problem/P5804）sort|greedy|brute_force|binary_search
P5963（https://www.luogu.com.cn/problem/P5963）greedy|sort
P6023（https://www.luogu.com.cn/problem/P6023）pointer|implemention|brain_teaser
P6243（https://www.luogu.com.cn/problem/P6243）greedy|custom_sort
P6179（https://www.luogu.com.cn/problem/P6179）greedy
P6380（https://www.luogu.com.cn/problem/P6380）greedy|implemention
P6446（https://www.luogu.com.cn/problem/P6446）greedy|implemention
P5019（https://www.luogu.com.cn/problem/P5019）greedy|implemention
P6462（https://www.luogu.com.cn/problem/P6462）greedy|classification_discussion
P6549（https://www.luogu.com.cn/problem/P6549）reverse_thinking|sort|implemention
P6785（https://www.luogu.com.cn/problem/P6785）brain_teaser|greedy|counter
P6851（https://www.luogu.com.cn/problem/P6851）greedy|implemention|sort
P7176（https://www.luogu.com.cn/problem/P7176）greedy|classical|maximum_property|observation
P7228（https://www.luogu.com.cn/problem/P7228）brain_teaser|greedy|tree_dfs
P7260（https://www.luogu.com.cn/problem/P7260）greedy|dp|operation
P7319（https://www.luogu.com.cn/problem/P7319）greedy|custom_sort
P7412（https://www.luogu.com.cn/problem/P7412）greedy
P7522（https://www.luogu.com.cn/problem/P7522）classification_discussion|greedy
P7633（https://www.luogu.com.cn/problem/P7633）euler_series|O(nlogn)|implemention|greedy
P7714（https://www.luogu.com.cn/problem/P7714）sort|prefix_max|pointer|counter
P7787（https://www.luogu.com.cn/problem/P7787）brain_teaser|tree|complete_binary_tree
P7813（https://www.luogu.com.cn/problem/P7813）greedy
P1031（https://www.luogu.com.cn/problem/P1031）greedy|card_split_average|classical
P2512（https://www.luogu.com.cn/problem/P2512）greedy|card_split_average|classical|circular_array
P1080（https://www.luogu.com.cn/problem/P1080）greedy|custom_sort
P1650（https://www.luogu.com.cn/problem/P1650）greedy|classical
P2088（https://www.luogu.com.cn/problem/P2088）greedy
P2816（https://www.luogu.com.cn/problem/P2816）sort|greedy|sorted_list
P3819（https://www.luogu.com.cn/problem/P3819）median|greedy
P3918（https://www.luogu.com.cn/problem/P3918）brain_teaser|greedy
P4025（https://www.luogu.com.cn/problem/P4025）greedy|custom_sort
P4266（https://www.luogu.com.cn/problem/P4266）greedy|implemention|suffix_max
P4447（https://www.luogu.com.cn/problem/P4447）greedy|deque
P4575（https://www.luogu.com.cn/problem/P4575）brain_teaser|state_compression
P4653（https://www.luogu.com.cn/problem/P4653）binary_search|pointer|greedy
P5093（https://www.luogu.com.cn/problem/P5093）brain_teaser|classical
P5425（https://www.luogu.com.cn/problem/P5425）mst_like|brain_teaser|greedy
P5884（https://www.luogu.com.cn/problem/P5884）brain_teaser
P5948（https://www.luogu.com.cn/problem/P5948）greedy|implemention
P6196（https://www.luogu.com.cn/problem/P6196）greedy
P6874（https://www.luogu.com.cn/problem/P6874）median_greedy
P8050（https://www.luogu.com.cn/problem/P8050）brain_teaser|coloring_method|operation
P7935（https://www.luogu.com.cn/problem/P7935）brain_teaser
P8109（https://www.luogu.com.cn/problem/P8109）sorted_list|greedy
P8669（https://www.luogu.com.cn/problem/P8669）greedy|maximum_mul
P8709（https://www.luogu.com.cn/problem/P8709）brain_teaser|implemention
P8732（https://www.luogu.com.cn/problem/P8732）greedy|brute_force|custom_sort
P8887（https://www.luogu.com.cn/problem/P8887）brain_teaser|greedy
P1342（https://www.luogu.com.cn/problem/P1342）brain_teaser|greedy
P1842（https://www.luogu.com.cn/problem/P1842）greedy
P2968（https://www.luogu.com.cn/problem/P2968）greedy|implemention|observation
P3619（https://www.luogu.com.cn/problem/P3619）greedy|classical|custom_sort
P3550（https://www.luogu.com.cn/problem/P3550）greedy|observation
P4823（https://www.luogu.com.cn/problem/P4823）greedy|regret_heapq|classical|brain_teaser
P4998（https://www.luogu.com.cn/problem/P4998）brain_teaser|greedy|prefix_sum
P5963（https://www.luogu.com.cn/problem/P5963）pair_wise|greedy|classical|custom_sort
P6002（https://www.luogu.com.cn/problem/P6002）brute_force|greedy|brain_teaser
P7148（https://www.luogu.com.cn/problem/P7148）greedy

===================================CodeForces===================================
1186D（https://codeforces.com/problemset/problem/1186/D）greedy|floor|property
792C（https://codeforces.com/contest/792/problem/C）classification_discussion|greedy
166E（https://codeforces.com/problemset/problem/166/E）implemention|brain_teaser|dp
1025C（https://codeforces.com/problemset/problem/1025/C）brain_teaser
1042C（https://codeforces.com/problemset/problem/1042/C）greedy|classification_discussion|implemention
439C（https://codeforces.com/problemset/problem/439/C）greedy|classification_discussion
1283E（https://codeforces.com/problemset/problem/1283/E）greedy|classification_discussion|linear_dp
1092C（https://codeforces.com/contest/1092/problem/C）brain_teaser|classification_discussion
1280B（https://codeforces.com/problemset/problem/1280/B）brain_teaser|classification_discussion
723C（https://codeforces.com/problemset/problem/723/C）greedy|implemention|construction
712C（https://codeforces.com/problemset/problem/712/C）reverse_thinking|implemention
747D（https://codeforces.com/problemset/problem/747/D）greedy|implemention
1148D（https://codeforces.com/problemset/problem/1148/D）greedy|custom_sort|construction
792C（https://codeforces.com/contest/792/problem/C）classification_discussion|greedy
830A（https://codeforces.com/problemset/problem/830/A）sort|greedy|action_scope
478C（https://codeforces.com/problemset/problem/478/C）greedy|math|property
1329A（https://codeforces.com/problemset/problem/1329/A）greedy|pointer|implemention
1401D（https://codeforces.com/problemset/problem/1401/D）greedy|dfs|brute_force|counter
600C（https://codeforces.com/problemset/problem/600/C）palindrome_substring|counter|greedy
1038D（https://codeforces.com/problemset/problem/1038/D）greedy|implemention|classification_discussion
349B（https://codeforces.com/problemset/problem/349/B）greedy|implemention
1370C（https://codeforces.com/problemset/problem/1370/C）greedy|implemention|winning_state
1822E（https://codeforces.com/contest/1822/problem/E）greedy|implemention|counter
1005E2（https://codeforces.com/contest/1005/problem/E2）median|inclusion_exclusion|prefix_sum|sorted_list|binary_search|LC2488
1512E（https://codeforces.com/contest/1512/problem/E）brain_teaser|greedy|big_to_small
1665C（https://codeforces.com/contest/1665/problem/C）graph|greedy
1649B（https://codeforces.com/contest/1649/problem/B）maximum_greedy|classical
1914E2（https://codeforces.com/contest/1914/problem/E2）greedy|custom_sort
1929D（https://codeforces.com/contest/1920/problem/D）data_range|brute_force|reverse_thinking
724D（https://codeforces.com/contest/724/problem/D）greedy|implemention|brain_teaser
1669D（https://codeforces.com/contest/1669/problem/D）brain_teaser
1807G2（https://codeforces.com/contest/1807/problem/G2）brain_teaser|classical|sorting|greedy
1873G（https://codeforces.com/contest/1873/problem/G）brain_teaser
977D（https://codeforces.com/contest/977/problem/D）brain_teaser|greedy|classical|sorting
978G（https://codeforces.com/contest/978/problem/G）brain_teaser|greedy|sorting|implemention|reverse_thinking
999D（https://codeforces.com/contest/999/problem/D）greedy|brute_force
1144G（https://codeforces.com/contest/1144/problem/G）linear_dp|greedy|classical|construction|brain_teaser
1157G（https://codeforces.com/contest/1157/problem/G）brain_teaser|brute_force|classical|implemention|greedy
1157F（https://codeforces.com/contest/1157/problem/F）greedy|brain_teaser|construction|specific_plan
1157C2（https://codeforces.com/contest/1157/problem/C2）greedy|brain_teaser|implemention
1183G（https://codeforces.com/contest/1183/problem/G）greedy|brain_teaser|implemention|classical
1183D（https://codeforces.com/contest/1183/problem/D）greedy|brain_teaser|implemention|classical
1183F（https://codeforces.com/contest/1183/problem/F）greedy|brain_teaser|classical|brute_force|special_judge
1203F1（https://codeforces.com/contest/1203/problem/F1）greedy|brain_teaser|linear_dp|define_sort|classical
1203F2（https://codeforces.com/contest/1203/problem/F2）greedy|brain_teaser|linear_dp|define_sort|classical
1249D2（https://codeforces.com/contest/1249/problem/D2）greedy|offline_query|sorted_list
1256F（https://codeforces.com/contest/1256/problem/F）greedy|brain_teaser|reverse_pair|bubble_sort|classical
1296E2（https://codeforces.com/contest/1296/problem/E2）greedy|brain_teaser|lis
1296E1（https://codeforces.com/contest/1296/problem/E1）greedy|brain_teaser|lis
1367F2（https://codeforces.com/contest/1367/problem/F2）greedy|brain_teaser|lis
1385F（https://codeforces.com/contest/1385/problem/F）greedy|brain_teaser|topological_sort
1399E2（https://codeforces.com/contest/1399/problem/E2）greedy|graph|brain_teaser|brute_force
1538G（https://codeforces.com/contest/1538/problem/G）greedy|brain_teaser
1512F（https://codeforces.com/contest/1512/problem/F）greedy|brain_teaser
1593G（https://codeforces.com/contest/1593/problem/G）greedy|brain_teaser|prefix_sum
1579E2（https://codeforces.com/contest/1579/problem/E2）greedy|brain_teaser
1560F2（https://codeforces.com/contest/1560/problem/F2）greedy
1660F2（https://codeforces.com/contest/1660/problem/F2）greedy|brain_teaser|sorted_list
1674E（https://codeforces.com/contest/1674/problem/E）greedy|brute_force
1772E（https://codeforces.com/contest/1772/problem/E）greedy|brain_teaser
1759G（https://codeforces.com/contest/1759/problem/G）greedy|brain_teaser
1883F（https://codeforces.com/contest/1883/problem/F）brain_teaser|prefix_suffix
1872G（https://codeforces.com/contest/1872/problem/G）brain_teaser|greedy
1899E（https://codeforces.com/contest/1899/problem/E）brain_teaser|greedy
1923D（https://codeforces.com/contest/1923/problem/D）brain_teaser|greedy|prefix_sum|binary_search
1923B（https://codeforces.com/contest/1923/problem/D）brain_teaser|implemention|greedy
1921E（https://codeforces.com/contest/1921/problem/E）brain_teaser|implemention|greedy|odd_even
1941F（https://codeforces.com/contest/1941/problem/F）brain_teaser|implemention|greedy|median|binary_search
1941C（https://codeforces.com/contest/1941/problem/C）brain_teaser|greedy
1974G（https://codeforces.com/contest/1974/problem/G）regret_heapq|implemention|brain_teaser|classical
1976B（https://codeforces.com/contest/1976/problem/B）brute_force|greedy
985C（https://codeforces.com/problemset/problem/985/C）greedy|brain_teaser|reverse_order
1978D（https://codeforces.com/contest/1978/problem/D）greedy|brain_teaser|implemention
1316C（https://codeforces.com/problemset/problem/1316/C）observation|math|brain_teaser
1156C（https://codeforces.com/problemset/problem/1156/C）greedy|two_pointers|classical|brain_teaser
1684D（https://codeforces.com/problemset/problem/1684/D）greedy|observation|contribution_method
1379C（https://codeforces.com/contest/1379/problem/C）observation|prefix_sum|binary_search|brute_force|greedy
1451D（https://codeforces.com/problemset/problem/1451/D）data_range|observation|classical|greedy|implemention
1295B（https://codeforces.com/problemset/problem/1295/B）observation|brain_teaser|classification_discussion
1870D（https://codeforces.com/problemset/problem/1870/D）observation|greedy|monotonic_stack
1415D（https://codeforces.com/problemset/problem/1415/D）observation|bit_operation|data_range
893D（https://codeforces.com/problemset/problem/893/D）greedy|brain_teaser|implemention|low_to_high
1849D（https://codeforces.com/problemset/problem/1849/D）observation|greedy|implemention
1496D（https://codeforces.com/problemset/problem/1496/D）observation|prefix_suffix|brute_force|greedy|implemention|game
1436D（https://codeforces.com/problemset/problem/1436/D）tree_bfs|greedy|brain_teaser
1621D（https://codeforces.com/problemset/problem/1621/D）greedy|brain_teaser
1700D（https://codeforces.com/problemset/problem/1700/D）greedy|brain_teaser
1430D（https://codeforces.com/problemset/problem/1430/D）greedy|two_pointers
1392D（https://codeforces.com/problemset/problem/1392/D）observation|brain_teaser
1238D（https://codeforces.com/problemset/problem/1238/D）observation
1186C（https://codeforces.com/problemset/problem/1186/C）observation|brain_teaser
372A（https://codeforces.com/problemset/problem/372/A）observation|greedy|classical
1804D（https://codeforces.com/problemset/problem/1804/D）greedy
282B（https://codeforces.com/problemset/problem/282/B）greedy
1257D（https://codeforces.com/problemset/problem/1257/D）suffix_max|greedy|implemention|classical
1539D（https://codeforces.com/problemset/problem/1539/D）greedy|two_pointers|implemention
865D（https://codeforces.com/problemset/problem/865/D）regret_heapq|greedy|classical
713C（https://codeforces.com/problemset/problem/713/C）greedy|brain_teaser|strictly_monotonic_trick|classical
13C（https://codeforces.com/problemset/problem/13/C）greedy|brain_teaser|regret_heapq|classical
1119E（https://codeforces.com/problemset/problem/1119/E）greedy
1515D（https://codeforces.com/problemset/problem/1515/D）greedy|brain_teaser|classification_discussion
1466D（https://codeforces.com/problemset/problem/1466/D）greedy|implemention
1282B2（https://codeforces.com/problemset/problem/1282/B2）greedy|linear_dp
2004D（https://codeforces.com/problemset/problem/2004/D）observation|data_range|brain_teaser|brute_force
468B（https://codeforces.com/problemset/problem/468/B）greedy|sort

====================================AtCoder=====================================
ARC062A（https://atcoder.jp/contests/abc046/tasks/arc062_a）brain_teaser|greedy|custom_sort
ARC088B（https://atcoder.jp/contests/abc083/tasks/arc088_b）brain_teaser|greedy
ABC116D（https://atcoder.jp/contests/abc116/tasks/abc116_d）brain_teaser|greedy
ABC137D（https://atcoder.jp/contests/abc137/tasks/abc137_d）reverse_order|brain_teaser|greedy
ABC333E（https://atcoder.jp/contests/abc333/tasks/abc333_e）reverse_order|greedy
ABC330F（https://atcoder.jp/contests/abc330/tasks/abc330_f）brain_teaser|greedy|brute_force|binary_search|prefix_sum
ABC314D（https://atcoder.jp/contests/abc314/tasks/abc314_d）reverse_order|brain_teaser
ABC313C（https://atcoder.jp/contests/abc313/tasks/abc313_c）brain_teaser|median_greedy
ABC310E（https://atcoder.jp/contests/abc310/tasks/abc310_e）brain_teaser|implemention
ABC308F（https://atcoder.jp/contests/abc308/tasks/abc308_f）brain_teaser|greedy
ABC296F（https://atcoder.jp/contests/abc296/tasks/abc296_f）brain_teaser|greedy|sorted_list|reverse_pair|property
ABC293F（https://atcoder.jp/contests/abc293/tasks/abc293_f）binary_search|brute_force|brain_teaser|classical
ABC290D（https://atcoder.jp/contests/abc290/tasks/abc290_d）brain_teaser|implemention|math
ABC347C（https://atcoder.jp/contests/abc347/tasks/abc347_c）brain_teaser|implemention
ABC347D（https://atcoder.jp/contests/abc347/tasks/abc347_d）greedy
ABC252F（https://atcoder.jp/contests/abc252/tasks/abc252_f）greedy|small_to_big|reverse_order|classical
ABC349D（https://atcoder.jp/contests/abc349/tasks/abc349_d）greedy|brain_teaser
ABC249F（https://atcoder.jp/contests/abc249/tasks/abc249_f）greedy|implemention|reverse_order|classical
ABC230D（https://atcoder.jp/contests/abc230/tasks/abc230_d）greedy
ABC229G（https://atcoder.jp/contests/abc229/tasks/abc229_g）implemention|median_greedy|two_pointers|classical|prefix_sum
ABC209C（https://atcoder.jp/contests/abc209/tasks/abc209_c）greedy|brain_teaser
ABC359F（https://atcoder.jp/contests/abc359/tasks/abc359_f）greedy|implemention|stack

=====================================AcWing=====================================
104（https://www.acwing.com/problem/content/106/）median|greedy
1536（https://www.acwing.com/problem/content/description/1538/）greedy|card_split_average|classical
105（https://www.acwing.com/problem/content/description/1538/）greedy|card_split_average|classical|circular_array
110（https://www.acwing.com/problem/content/112/）greedy
123（https://www.acwing.com/problem/content/description/125/）median_greedy
125（https://www.acwing.com/problem/content/127/）greedy|custom_sort
127（https://www.acwing.com/problem/content/description/129/）partial_order|sort|greedy
145（https://www.acwing.com/problem/content/147/）heapq|greedy
122（https://www.acwing.com/problem/content/124/）greedy|card_split_average|classical|circular_array
4204（https://www.acwing.com/problem/content/description/4207/）construction
4307（https://www.acwing.com/problem/content/description/4310/）lexicographical_order|brute_force|greedy
4313（https://www.acwing.com/problem/content/4316/）full_binary_tree|tree_dp|greedy|LC2673
4426（https://www.acwing.com/problem/content/4429/）brain_teaser|brain_teaser|property
4427（https://www.acwing.com/problem/content/4430/）greedy|construction
4429（https://www.acwing.com/problem/content/description/4432/）greedy|custom_sort|prefix_suffix|brute_force
4430（https://www.acwing.com/problem/content/description/4433/）brute_force|prefix_suffix|bracket
4492（https://www.acwing.com/problem/content/description/4495/）brain_teaser|odd_even
4623（https://www.acwing.com/problem/content/description/4626/）greedy|implemention


=====================================CodeChef=====================================
1（https://www.codechef.com/problems/CHANGEXY）greedy|implemention
2（https://www.codechef.com/problems/DESTBRIDGE2）greedy|implemention

"""
import bisect
import heapq
import math
from bisect import insort_left, bisect_left
from collections import Counter, deque, defaultdict
from functools import reduce
from heapq import heappop, heapify, heappush
from typing import List

from src.basis.binary_search.template import BinarySearch
from src.data_structure.sorted_list.template import SortedList
from src.mathmatics.number_theory.template import NumFactor
from src.utils.fast_io import FastIO


class Solution:
    def __int__(self):
        return

    @staticmethod
    def cf_1005e2(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1005/problem/E2
        tag: median|inclusion_exclusion|prefix_sum|sorted_list|binary_search|LC2488|brain_teaser
        """

        n, m = ac.read_list_ints()
        nums = ac.read_list_ints()

        def check(x):
            cur = res = s = 0
            dct = defaultdict(int)
            dct[cur ^ ac.random_seed] = 1
            for num in nums:
                if num >= x:
                    s += dct[cur ^ ac.random_seed]
                    cur += 1
                else:
                    cur -= 1
                    s -= dct[cur ^ ac.random_seed]
                res += s
                dct[cur ^ ac.random_seed] += 1
            return res

        ac.st(check(m) - check(m + 1))
        return

    @staticmethod
    def cf_1038d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1038/D
        tag: greedy|implemention|classification_discussion
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        if n == 1:
            ac.st(nums[0])
            return
        if all(num > 0 for num in nums):
            ac.st(sum(nums) - 2 * min(nums))
        elif all(num < 0 for num in nums):
            ac.st(sum(-num for num in nums) + 2 * max(nums))
        else:
            ac.st(sum(abs(num) for num in nums))
        return

    @staticmethod
    def lg_p2512(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2512
        tag: greedy|card_split_average|classical|circular_array
        """
        # 环形均分纸牌问题
        n = ac.read_int()
        nums = [ac.read_int() for _ in range(n)]
        m = sum(nums) // n
        x = 0
        pre = []
        for i in range(n):
            x += m - nums[i]
            pre.append(x)
        pre.sort()
        y = pre[n // 2]
        ans = sum(abs(num - y) for num in pre)
        ac.st(ans)
        return

    @staticmethod
    def abc_46b(ac=FastIO()):
        # brain_teaser|，不等式greedy
        n = ac.read_int()
        a = b = 1
        for _ in range(n):
            x, y = ac.read_list_ints()
            z1 = a // x + int(a % x > 0)
            z2 = b // y + int(b % y > 0)
            z = max(z1, z2)
            a = z * x
            b = z * y
        ac.st(a + b)
        return

    @staticmethod
    def ac_105(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/1538/
        tag: greedy|card_split_average|classical|circular_array
        """

        def check(nums):
            # 环形均分纸牌
            nn = len(nums)
            s = sum(nums)
            if s % nn:
                return -1
            mm = s // nn
            x = 0
            pre = []
            for i in range(nn):
                x += mm - nums[i]
                pre.append(x)
            pre.sort()
            y = pre[nn // 2]
            ans = sum(abs(num - y) for num in pre)
            return ans

        m, n, t = ac.read_list_ints()
        row = [0] * m
        col = [0] * n
        for _ in range(t):
            xx, yy = ac.read_list_ints_minus_one()
            row[xx] += 1
            col[yy] += 1
        ans1 = check(row)
        ans2 = check(col)
        if ans1 != -1 and ans2 != -1:
            ac.lst(["both", ans1 + ans2])
        elif ans1 != -1:
            ac.lst(["row", ans1])
        elif ans2 != -1:
            ac.lst(["column", ans2])
        else:
            ac.st("impossible")
        return

    @staticmethod
    def ac_123(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/125/
        tag: median_greedy
        """
        # mediangreedy扩展问题，连续相邻sorting减去下标后再sorting
        n = ac.read_int()
        lst_x = []
        lst_y = []
        for _ in range(n):
            x, y = ac.read_list_ints()
            lst_y.append(y)
            lst_x.append(x)
        lst_y.sort()
        mid = lst_y[n // 2]
        ans = sum(abs(pos - mid) for pos in lst_y)

        lst_x.sort()
        lst_x = [lst_x[i] - i for i in range(n)]
        lst_x.sort()
        mid = lst_x[n // 2]
        ans += sum(abs(pos - mid) for pos in lst_x)
        ac.st(ans)
        return

    @staticmethod
    def ac_125(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/127/
        tag: greedy|custom_sort
        """
        # greedy思路，邻项交换
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: it[0] + it[1])
        ans = -math.inf
        pre = 0
        for w, s in nums:
            ans = max(ans, pre - s)
            pre += w
        ac.st(ans)
        return

    @staticmethod
    def ac_127(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/129/
        tag: partial_order|sort|greedy
        """
        # 二维sortinggreedy
        n, m = ac.read_list_ints()
        machine = [ac.read_list_ints() for _ in range(n)]
        task = [ac.read_list_ints() for _ in range(m)]
        machine.sort(reverse=True)
        task.sort(reverse=True)
        lst = []
        ans = money = j = 0
        for i in range(m):
            tm, level = task[i]
            while j < n and machine[j][0] >= tm:
                insort_left(lst, machine[j][1])
                j += 1
            ind = bisect_left(lst, level)
            if ind < len(lst):
                lst.pop(ind)
                ans += 1
                money += 500 * tm + 2 * level
        ac.lst([ans, money])
        return

    @staticmethod
    def ac_145(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/147/
        tag: heapq|greedy
        """
        # heapq|greedy
        lst = []
        cnt = 0
        while cnt < 10000:
            cur = ac.read_list_ints()
            if not cur:
                cnt += 1
            lst.extend(cur)

        lst = deque(lst)
        while lst:
            n = lst.popleft()
            cur = []
            for _ in range(n):
                p, d = lst.popleft(), lst.popleft()
                cur.append([p, d])
            cur.sort(key=lambda it: it[1])
            stack = []
            for p, d in cur:
                heapq.heappush(stack, p)
                if len(stack) == d + 1:
                    heapq.heappop(stack)
            ac.st(sum(stack))
        return

    @staticmethod
    def lc_2745(x: int, y: int, z: int) -> int:
        """
        url: https://leetcode.cn/problems/construct-the-longest-new-string/
        tag: brain_teaser|greedy
        """
        # brain_teasergreedybrain_teaser|
        return z * 2 + min(x, y) * 4 + int((max(x, y) - min(x, y)) > 0) * 2

    @staticmethod
    def lg_p1080(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1080
        tag: greedy|custom_sort
        """
        # greedy，举例两项确定sorting公式
        n = ac.read_int()
        a, b = ac.read_list_ints()
        lst = [ac.read_list_ints() for _ in range(n)]
        lst.sort(key=lambda x: x[0] * x[1] - x[1])
        ans = 0
        pre = a
        for a, b in lst:
            ans = max(ans, pre // b)
            pre *= a
        ac.st(ans)
        return

    @staticmethod
    def lg_p1650(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1650
        tag: greedy|classical
        """
        # greedy，优先上对上其次下对下最后下对上
        ac.read_int()
        a = deque(sorted(ac.read_list_ints(), reverse=True))
        b = deque(sorted(ac.read_list_ints(), reverse=True))

        ans = 0
        while a and b:
            # 上对上
            if a[0] > b[0]:
                a.popleft()
                b.popleft()
                ans += 200
            # 下对下
            elif a[-1] > b[-1]:
                a.pop()
                b.pop()
                ans += 200
            # 下对上
            else:
                x = a.pop()
                y = b.popleft()
                if x > y:
                    ans += 200
                elif x < y:
                    ans -= 200
        ac.st(ans)
        return

    @staticmethod
    def lg_p2088(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2088
        tag: greedy
        """
        # 队列集合greedy，取空闲的，或者下一个离得最远的
        ans = 0
        k, n = ac.read_list_ints()
        nums = []
        while len(nums) < n:
            nums.extend(ac.read_list_ints())

        busy = set()
        post = defaultdict(deque)
        for i in range(n):
            post[nums[i]].append(i)
        for x in post:
            post[x].append(n)

        for i in range(n):
            if nums[i] in busy:
                continue
            if len(busy) < k:
                busy.add(nums[i])
                continue
            nex = -1
            for x in busy:
                while post[x] and post[x][0] < i:
                    post[x].popleft()
                if nex == -1 or post[x][0] >= post[nex][0]:
                    nex = x
            busy.discard(nex)
            busy.add(nums[i])
            ans += 1
        ac.st(ans)
        return

    @staticmethod
    def lg_p2816(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2816
        tag: sort|greedy|sorted_list
        """
        # sorting后从小到大greedy放置，STL维护当前积木列高度
        lst = SortedList()
        ac.read_int()
        nums = ac.read_list_ints()
        nums.sort()
        for num in nums:
            i = lst.bisect_left(num)
            if (0 <= i < len(lst) and lst[i] > num) or i == len(lst):
                i -= 1
            if 0 <= i < len(lst):
                lst.add(lst.pop(i) + 1)
            else:
                lst.add(1)
        ac.st(len(lst))
        return

    @staticmethod
    def lg_p3819(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P3819
        tag: median|greedy
        """
        # mediangreedy题
        length, n = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(n)]
        s = sum(x for _, x in nums)
        nums.sort()
        pre = 0
        for pos, x in nums:
            pre += x
            if pre >= s // 2:
                ans = sum(abs(x - pos) * y for x, y in nums)
                ac.st(ans)
                break
        return

    @staticmethod
    def lg_p4025(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4025
        tag: greedy|custom_sort
        """
        # greedy血量与增幅custom_sort
        n, z = ac.read_list_ints()
        pos = []
        neg = []
        for i in range(n):
            d, a = ac.read_list_ints()
            if d < a:
                pos.append([d, a, i])
            else:
                neg.append([d, a, i])
        pos.sort(key=lambda it: it[0])
        neg.sort(key=lambda it: -it[1])
        ans = []
        for d, a, i in pos + neg:
            z -= d
            if z <= 0:
                ac.st("NIE")
                return
            z += a
            ans.append(i + 1)
        ac.st("TAK")
        ac.lst(ans)
        return

    @staticmethod
    def lg_p4266(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4266
        tag: greedy|implemention|suffix_max
        """
        # 后缀最大值greedyimplemention
        length, n, rf, rb = ac.read_list_ints()
        nums = [[0, 0]] + [ac.read_list_ints() for _ in range(n)]
        n += 1
        # 记录后缀最大值序列
        post = [n - 1] * n
        ind = n - 1
        for i in range(n - 2, -1, -1):
            post[i] = ind
            if nums[i][1] > nums[ind][1]:
                ind = i
        path = [post[0]]
        while post[path[-1]] != path[-1]:
            path.append(post[path[-1]])
        # implemention
        ans = t = pre = 0
        for i in path:
            cur = nums[i][0]
            c = nums[i][1]
            ans += (cur - pre) * (rf - rb) * c
            t += (cur - pre) * rf
            pre = cur
        ac.st(ans)
        return

    @staticmethod
    def lg_p4447(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4447
        tag: greedy|deque
        """
        # greedy队列使得连续值序列最少的分组长度最大
        ac.read_int()
        lst = ac.read_list_ints()
        lst.sort()
        # 记录末尾值为 num 的连续子序列长度
        cnt = defaultdict(list)
        for num in lst:
            if cnt[num - 1]:
                # 将最小的长度取出添上
                val = heapq.heappop(cnt[num - 1])
                heapq.heappush(cnt[num], val + 1)
            else:
                # 单独成为一个序列
                heapq.heappush(cnt[num], 1)
        ac.st(min(min(cnt[k]) for k in cnt if cnt[k]))
        return

    @staticmethod
    def lg_p4575(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4575
        tag: brain_teaser|state_compression
        """
        # brain_teaser|状压运算
        for _ in range(ac.read_int()):
            m = ac.read_int()
            k = ac.read_int()
            dct = [set() for _ in range(m)]
            for _ in range(k):
                i, j = ac.read_list_ints()
                dct[i].add(j)

            dp = [sum((1 << j) for j in dct[i]) for i in range(m)]
            ans = True
            for i in range(m):
                if not ans:
                    break
                for j in range(i + 1, m):
                    if dp[i] & dp[j] and dp[i] ^ dp[j]:
                        ans = False
                        break
            ac.st("Yes" if ans else "No")
        return

    @staticmethod
    def lg_p4653(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4653
        tag: binary_search|pointer|greedy
        """
        # 看似binary_searchpointergreedy选取
        n = ac.read_int()
        nums1 = []
        nums2 = []
        for _ in range(n):
            x, y = ac.read_list_floats()
            nums1.append(x)
            nums2.append(y)
        nums1.sort(reverse=True)
        nums2.sort(reverse=True)
        # two_pointers选择
        ans = i = j = a = b = 0
        light_a = light_b = 0
        while i < n or j < n:
            if i < n and (a - light_b < b - light_a or j == n):
                a += nums1[i] - 1
                i += 1
                light_a += 1
            else:
                b += nums2[j] - 1
                j += 1
                light_b += 1
            ans = max(ans, min(a - light_b, b - light_a))
        ac.st("%.4f" % ans)
        return

    @staticmethod
    def lg_p5093(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5093
        tag: brain_teaser|classical
        """
        # brain_teaser集合确定轮数
        n, k = ac.read_list_ints()
        nums = [ac.read_int() for _ in range(n)]
        pre = set()
        ans = 1
        for num in nums:
            pre.add(num)
            if len(pre) == k:
                ans += 1
                pre = set()
        ac.st(ans)
        return

    @staticmethod
    def lg_p5425(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5425
        tag: mst_like|brain_teaser|greedy
        """
        # 看似mst|，实则brain_teasergreedy距离
        n, k = ac.read_list_ints()
        ans = (2019201913 * (k - 1) + 2019201949 * n) % 2019201997
        ac.st(ans)
        return

    @staticmethod
    def lg_p5884(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P5884
        tag: brain_teaser|reverse_order|mst
        """
        n = ac.read_int()
        last = [0] * n
        edges = []
        for x in range(n * (n - 1) // 2):
            i, j = ac.read_list_ints()
            if i > j:
                i, j = j, i
            last[i] = x
            edges.append(i*n+j)
        for x, val in enumerate(edges):
            i, j = val//n, val%n
            if last[i] == x:
                ac.st(1)
            else:
                ac.st(0)
        return

    @staticmethod
    def lg_p6196(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6196
        tag: greedy
        """
        # greedy 1 分段代价
        ac.read_int()
        nums = ac.read_list_ints()
        ans = 0
        lst = []
        for num in nums:
            if num == 1:
                if lst:
                    m = len(lst)
                    ans += min(lst)
                    for i in range(1, m):
                        ans += lst[i - 1] * lst[i]
                ans += 1
                lst = []
            else:
                lst.append(num)
        if lst:
            m = len(lst)
            ans += min(lst)
            for i in range(1, m):
                ans += lst[i - 1] * lst[i]
        ac.st(ans)
        return

    @staticmethod
    def lg_p6874(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6874
        tag: median_greedy|brain_teaser
        """
        n = ac.read_int()  # MLE
        a = ac.read_list_ints()
        b = ac.read_list_ints()
        for i in range(n):
            w = abs(i - n // 2)
            a[i] -= w
            b[i] -= w
        a.extend(b)
        del b
        a.sort()
        x = max(0, a[n])
        ac.st(sum(abs(x - num) for num in a))
        return

    @staticmethod
    def lg_p8050(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8050
        tag: brain_teaser|coloring_method|operation
        """
        # brain_teaser黑白coloring_method任意操作不改变黑白元素和的差值
        m1, n1, m2, n2, k = ac.read_list_ints()
        black = white = cnt = state = 0
        for i in range(m1 + m2):
            lst = ac.read_list_ints()
            for j in range(len(lst)):
                if lst[j] != 999999:
                    if (i + j) % 2:
                        black += lst[j]
                        cnt += 1
                    else:
                        white += lst[j]
                else:
                    state = (i + j) % 2
                    cnt += (i + j) % 2
        ans = (2 * cnt - m1 * n1 - m2 * n2) * k - black + white
        if not state:
            ans = -ans
        ac.st(ans)
        return

    @staticmethod
    def lg_p8732(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P8732
        tag: greedy|brute_force|custom_sort
        """
        # greedybrute_force两项优先级公式
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: sum(it))
        ans = pre = 0
        for s, a, e in nums:
            pre += s + a
            ans += pre
            pre += e
        ac.st(ans)
        return

    @staticmethod
    def ac_4307(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4310/
        tag: lexicographical_order|brute_force|greedy
        """
        # lexicographical_orderbrute_forcegreedy
        a = [int(w) for w in str(ac.read_int())]
        b = [int(w) for w in str(ac.read_int())]
        a.sort()
        if len(a) < len(b):
            res = a[::-1]
        else:
            n = len(a)
            res = []
            for x in range(n):
                for i in range(n - 1 - x, -1, -1):
                    tmp = a[:]
                    tmp.pop(i)
                    if res + [a[i]] + tmp <= b:
                        res.append(a[i])
                        a.pop(i)
                        break
        ac.st("".join(str(x) for x in res))
        return

    @staticmethod
    def ac_4313(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4316/
        tag: full_binary_tree|tree_dp|greedy|LC2673
        """
        # 满二叉树tree_dpgreedy
        n = ac.read_int()
        m = 2 ** (n + 1)
        dp = [0] * m
        nums = ac.read_list_ints()
        ans = 0
        for i in range(m // 2 - 1, 0, -1):
            left = dp[i * 2] + nums[i * 2 - 2]
            right = dp[i * 2 + 1] + nums[i * 2 - 1]
            x = max(left, right)
            dp[i] = x
            ans += x * 2 - left - right
        ac.st(ans)
        return

    @staticmethod
    def ac_4426(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4429/
        tag: brain_teaser|brain_teaser|property
        """
        # brain_teaser|brain_teaser，等价于末尾两位数字可以被4整除
        s = ac.read_str()
        ans = 0
        n = len(s)
        for i in range(n):
            if i - 1 >= 0 and int(s[i - 1:i + 1]) % 4 == 0:
                ans += i  # 两位数及以上
            if int(s[i]) % 4 == 0:
                ans += 1  # 一位数
        ac.st(ans)
        return

    @staticmethod
    def ac_4427(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/4430/
        tag: greedy|construction
        """
        # 树形greedyconstruction
        n = ac.read_int()
        dct = [[] for _ in range(n)]
        parent = ac.read_list_ints_minus_one()
        for i in range(n - 1):
            dct[parent[i]].append(i + 1)
        s = ac.read_list_ints()
        ans = [0] * n

        stack = [[0, 0, 0]]
        while stack:
            x, pre, ss = stack.pop()
            pre += 1
            if pre % 2:  # 奇数位没得选
                ans[x] = s[x] - ss
            else:
                lst = []  # 偶数位greedy取最大值
                for y in dct[x]:
                    lst.append(s[y])
                if lst:
                    ans[x] = min(lst) - ss

            for y in dct[x]:
                stack.append([y, pre, ss + ans[x]])

        ac.st(sum(ans) if all(x >= 0 for x in ans) else -1)
        return

    @staticmethod
    def ac_4429(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4432/
        tag: greedy|custom_sort|prefix_suffix|brute_force
        """
        # 邻项公式greedysorting，prefix_suffixbrute_force
        n, x1, y1, x2, y2 = ac.read_list_ints()
        pos = [ac.read_list_ints() for _ in range(n)]
        dis1 = [(x - x1) * (x - x1) + (y - y1) * (y - y1) for x, y in pos]
        dis2 = [(x - x2) * (x - x2) + (y - y2) * (y - y2) for x, y in pos]
        # sorting
        ind = list(range(n))
        ind.sort(key=lambda it: dis1[it] - dis2[it])
        # 后缀最大值
        post = [0] * (n + 1)
        ceil = 0
        for i in range(n - 1, -1, -1):
            ceil = max(dis2[ind[i]], ceil)
            post[i] = ceil

        # brute_force前缀
        ans = post[0]
        pre = 0
        for i in range(n):
            pre = max(pre, dis1[ind[i]])
            if pre + post[i + 1] < ans:
                ans = pre + post[i + 1]
        ac.st(ans)
        return

    @staticmethod
    def ac_4430(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4433/
        tag: brute_force|prefix_suffix|bracket
        """
        # 括号匹配brute_force，prefix_suffix遍历
        n = ac.read_int()
        s = ac.read_str()
        ans = 0

        # 左变右
        post = [-1] * (n + 1)
        post[n] = 0
        right = 0
        for i in range(n - 1, -1, -1):
            if s[i] == ")":
                right += 1
            else:
                if not right:
                    break
                right -= 1
            post[i] = right

        left = 0
        for i in range(n):
            if s[i] == ")":
                if not left:
                    break
                left -= 1
            else:
                if post[i + 1] + 1 == left and post[i + 1] != -1:
                    ans += 1
                left += 1

        # 右变左
        pre = [-1] * (n + 1)
        pre[0] = 0
        left = 0
        for i in range(n):
            if s[i] == "(":
                left += 1
            else:
                if not left:
                    break
                left -= 1
            pre[i + 1] = left

        right = 0
        for i in range(n - 1, -1, -1):
            if s[i] == "(":
                if not right:
                    break
                right -= 1
            else:
                if pre[i] + 1 == right and pre[i] != -1:
                    ans += 1
                right += 1
        ac.st(ans)
        return

    @staticmethod
    def ac_4492(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4495/
        tag: brain_teaser|odd_even
        """
        # brain_teaser分为奇数与偶数讨论
        n = ac.read_int()
        if n % 2 == 0:
            ac.st(n // 2)
        else:
            lst = NumFactor().get_prime_factor(n)
            x = lst[0][0]
            ac.st(1 + (n - x) // 2)
        return

    @staticmethod
    def ac_4623(ac=FastIO()):
        """
        url: https://www.acwing.com/problem/content/description/4626/
        tag: greedy|implemention
        """
        # greedyimplemention
        n, t = ac.read_list_ints()
        a = ac.read_list_ints()
        ans = 0
        while a:
            s = sum(a)
            ans += (t // s) * len(a)
            t %= s
            b = []
            for num in a:
                if t >= num:
                    t -= num
                    ans += 1
                    b.append(num)
            a = b[:]
        ac.st(ans)
        return

    @staticmethod
    def cf_1665c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1665/problem/C
        tag: graph|greedy
        """

        def solve():
            n = ac.read_int()
            dct = [[] for _ in range(n)]
            parent = ac.read_list_ints_minus_one()
            for i in range(n - 1):
                dct[parent[i]].append(i + 1)
            lst = [len(dct[i]) for i in range(n) if dct[i]] + [1]
            lst.sort(reverse=True)
            m = len(lst)
            ans = m
            for i in range(m):
                lst[i] -= m - i
            stack = [-x for x in lst if x > 0]
            heapq.heapify(stack)
            cnt = 0
            while stack and -stack[0] > cnt:
                cnt += 1
                x = heapq.heappop(stack)
                x += 1
                if x:
                    heapq.heappush(stack, x)
            ans += cnt
            ac.st(ans)
            return

        for _ in range(ac.read_int()):
            solve()
        return

    @staticmethod
    def lc_858(p: int, q: int) -> int:
        """
        url: https://leetcode.cn/problems/mirror-reflection/description/
        tag: brain_teaser
        """
        # brain_teaserbrain_teaser|

        g = math.gcd(p, q)
        # 求解等式 k*p = m*q

        # 左右次数k合计为偶数
        k = p // g
        if k % 2 == 0:
            return 2

        # 上下次数m合计为偶数
        m = q // g
        if m % 2 == 0:
            return 0
        return 1

    @staticmethod
    def lc_991(start: int, target: int) -> int:
        """
        url: https://leetcode.cn/problems/broken-calculator/
        tag: reverse_order|reverse_thinking|greedy|odd_even|implemention
        """
        # 逆向greedy，偶数除2奇数|1
        ans = 0
        while target > start:
            if target % 2:
                target += 1
            else:
                target //= 2
            ans += 1
        return ans + start - target

    @staticmethod
    def lc_1503(n: int, left: List[int], right: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/last-moment-before-all-ants-fall-out-of-a-plank/
        tag: brain_teaser|classical
        """
        # 
        ans = 0
        for x in left:
            if x > ans:
                ans = x
        for x in right:
            if n - x > ans:
                ans = n - x
        return ans

    @staticmethod
    def lc_1675(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimize-deviation-in-array/
        tag: brain_teaser|greedy
        """
        # brain_teaserbrain_teaser|greedy
        lst = SortedList([num if num % 2 == 0 else num * 2 for num in nums])
        ans = lst[-1] - lst[0]
        while True:
            cur = lst[-1] - lst[0]
            ans = ans if ans < cur else cur
            if lst[-1] % 2:
                break
            lst.add(lst.pop() // 2)
        return ans

    @staticmethod
    def lc_1808(prime_factors: int) -> int:
        """
        url: https://leetcode.cn/problems/maximize-number-of-nice-divisors/
        tag: mod|greedy|classical|maximum_mul
        """
        # 按照模3的因子个数greedy处理，将和拆分成最大乘积
        mod = 10 ** 9 + 7
        if prime_factors <= 2:
            return prime_factors
        if prime_factors % 3 == 0:
            return pow(3, prime_factors // 3, mod)
        elif prime_factors % 3 == 1:
            return (4 * pow(3, prime_factors // 3 - 1, mod)) % mod
        else:
            return (2 * pow(3, prime_factors // 3, mod)) % mod

    @staticmethod
    def lc_1927(num: str) -> bool:
        """
        url: https://leetcode.cn/problems/sum-game/description/
        tag: game_dp|brain_teaser|classification_discussion
        """

        # 博弈brain_teaser|classification_discussion
        def check(s):
            res = 0
            cnt = 0
            for w in s:
                if w.isnumeric():
                    res += int(w)
                else:
                    cnt += 1
            return [res, cnt]

        # 左右两边的数字和以及问号个数
        n = len(num)
        a, x = check(num[:n // 2])
        b, y = check(num[n // 2:])

        # Alice把宝压在右边
        b_add = 9 * (y // 2 + y % 2)
        if y % 2 == 0:
            a_add = 9 * (x // 2)
        else:
            a_add = 9 * (x // 2 + x % 2)
        if a + a_add < b + b_add:
            return True

        # Alice把宝压在左边
        a_add = 9 * (x // 2 + x % 2)
        if x % 2 == 0:
            b_add = 9 * (y // 2)
        else:
            b_add = 9 * (y // 2 + y % 2)
        if b + b_add < a + a_add:
            return True

        # 左右都不能获胜
        return False

    @staticmethod
    def lc_2592(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/maximize-greatness-of-an-array/
        tag: classical|greedy|sort|two_pointers
        """
        # classicalgreedysorting后two_pointers
        n = len(nums)
        nums.sort()
        j = 0
        ans = 0
        for i in range(n):
            while j < n and nums[i] >= nums[j]:
                j += 1
            if j < n:
                ans += 1
                j += 1
        return ans

    @staticmethod
    def lc_2568(nums: List[int]) -> int:
        """
        url: https://leetcode.cn/problems/minimum-impossible-or/
        tag: brain_teaser|greedy|guess|watch_pattern
        """
        # brain_teasergreedy，可以根据打表观察规律
        dct = set(nums)
        for i in range(34):
            if 1 << i not in dct:
                return 1 << i
        return -1

    @staticmethod
    def lg_p1286(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P1286
        tag: brain_teaser|sorted_list
        """
        while True:
            nums = ac.read_list_strs()
            if not nums:
                break
            n = int(nums[0])
            nums = sorted([int(x) for x in nums[1:]])
            s = sum(nums) // (n - 1)

            def check():
                ans = [a1]
                lst = SortedList(nums)
                for _ in range(n - 1):
                    x = lst[0] - ans[0]
                    if x >= ans[-1]:
                        ans.append(x)
                        for num in ans[:-1]:
                            j = lst.bisect_left(num + x)
                            if not (0 <= j < len(lst) and lst[j] == num + x):
                                return []
                            lst.pop(j)
                    else:
                        return []
                return ans

            for a1 in range(0, s // n + 1):
                res = check()
                if res:
                    ac.lst(res)
                    break
            else:
                ac.st("Impossible")
        return

    @staticmethod
    def cf_1929d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1920/problem/D
        tag: data_range|brute_force|reverse_thinking
        """
        ceil = 10 ** 18
        for _ in range(ac.read_int()):
            ac.get_random_seed()
            n, q = ac.read_list_ints()
            nums = [ac.read_list_ints() for _ in range(n)]
            queries = ac.read_list_ints()
            c = 0
            ops = []
            for i in range(n):
                b, x = nums[i]
                if b == 2:
                    ops.append([2, x + 1, i, i])
                    c *= (x + 1)
                else:
                    if not ops or ops[-1][0] != 1:
                        ops.append([1, 1, i, i])
                    else:
                        ops[-1][1] += 1
                        ops[-1][-1] = i
                    c += 1
                if c > ceil:
                    break

            ans = [0] * q
            m = len(ops)
            for i, kk in enumerate(queries):

                cc = c
                for ii in range(m - 1, -1, -1):
                    bb, xx, ss, tt = ops[ii]
                    if bb == 2:
                        if cc // xx > kk:
                            cc //= xx
                        else:
                            kk = kk % (cc // xx)
                            if kk == 0:
                                kk = (cc // xx)
                            cc //= xx
                    else:
                        if cc - xx >= kk:
                            cc -= xx
                        else:
                            cc -= xx
                            ans[i] = nums[ss + kk - cc - 1][1]
                            break
            ac.lst(ans)
        return

    @staticmethod
    def lc_3012(nums: List[int]) -> int:
        """
        url: https://leetcode.com/problems/minimize-length-of-array-using-operations/
        tag: brain_teaser|perishu_theorem|hard|greedy
        """
        low = min(nums)
        gcd = reduce(math.gcd, nums)
        if gcd < low:
            return 1
        cnt = nums.count(low)
        return (cnt + 1) // 2

    @staticmethod
    def cf_724d(ac=FastIO()):
        """
        url: https://codeforces.com/contest/724/problem/D
        tag: greedy|implemention|brain_teaser
        """
        m = ac.read_int()
        s = ac.read_str()
        n = len(s)
        for i in range(26):
            ind = []
            w = chr(i + ord("a"))
            diff = [0] * n
            for j in range(n):
                if ord(s[j]) - ord("a") <= i:
                    ind.append(j)
                    diff[j] = 1
            pre = ac.accumulate(diff)
            if all(pre[i + 1] - pre[i - m + 1] > 0 for i in range(m - 1, n)):
                stack = [-1]
                for j in ind:
                    while len(stack) >= 2 and s[stack[-1]] == w and j - stack[-2] <= m:
                        stack.pop()
                    stack.append(j)
                while len(stack) >= 2 and s[stack[-1]] == w and stack[-2] >= n - m:
                    stack.pop()
                lst = [s[x] for x in stack[1:]]
                lst.sort()
                ac.st("".join(lst))
                return
        return

    @staticmethod
    def cf_1144g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1144/problem/G
        tag: linear_dp|greedy|classical|construction|brain_teaser
        """
        ascend = -math.inf
        descend = math.inf
        n = ac.read_int()
        nums = ac.read_list_ints()
        ans = [0] * n
        for i in range(n):
            if ascend < nums[i] and descend <= nums[i]:
                ascend = nums[i]
            elif ascend >= nums[i] and descend > nums[i]:
                descend = nums[i]
                ans[i] = 1
            elif ascend < nums[i] < descend:
                if i + 1 < n and nums[i + 1] < nums[i]:
                    ans[i] = 1
                    descend = nums[i]
                else:
                    ascend = nums[i]
            else:
                ac.no()
                break
        else:
            ac.yes()
            ac.lst(ans)
        return

    @staticmethod
    def cf_1157g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1157/problem/G
        tag: brain_teaser|brute_force|classical|implemention|greedy
        """
        m, n = ac.read_list_ints()
        grid = [ac.read_list_ints() for _ in range(m)]

        tmp = [g[:] for g in grid]
        row = [0] * m
        col = [0] * n
        for j in range(n):  # first_row = 0
            if tmp[0][j]:
                col[j] = 1
                for i in range(m):
                    tmp[i][j] = 1 - tmp[i][j]

        cnt = 0
        for i in range(1, m):
            if cnt >= 2:
                break
            dct = set(tmp[i])
            if len(dct) == 1:
                if cnt:
                    if dct != {1}:
                        row[i] = 1
                else:
                    if dct != {0}:
                        row[i] = 1
                continue
            pre = tmp[i][0]
            cur = 1
            for num in tmp[i][1:]:
                if num != pre:
                    cur += 1
                pre = num
                if cur > 2:
                    break
            if cur > 2:
                cnt = 2
            else:
                if tmp[i][0]:
                    row[i] = 1
                cnt += 1
        if cnt <= 1:
            ac.yes()
            ac.st("".join(str(x) for x in row))
            ac.st("".join(str(x) for x in col))
            return

        tmp = [g[:] for g in grid]  # last_row = 1
        row = [0] * m
        col = [0] * n
        for j in range(n):
            if tmp[-1][j] == 0:
                col[j] = 1
                for i in range(m):
                    tmp[i][j] = 1 - tmp[i][j]

        cnt = 0
        for i in range(m - 2, -1, -1):
            if cnt >= 2:
                break
            dct = set(tmp[i])
            if len(dct) == 1:
                if cnt:
                    if dct != {0}:
                        row[i] = 1
                else:
                    if dct != {1}:
                        row[i] = 1
                continue
            pre = tmp[i][0]
            cur = 1
            for num in tmp[i][1:]:
                if num != pre:
                    cur += 1
                pre = num
                if cur > 2:
                    break
            if cur > 2:
                cnt = 2
            else:
                if tmp[i][0]:
                    row[i] = 1
                cnt += 1
        if cnt <= 1:
            ac.yes()
            ac.st("".join(str(x) for x in row))
            ac.st("".join(str(x) for x in col))
            return
        ac.no()
        return

    @staticmethod
    def cf_1157f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1157/problem/F
        tag: greedy|brain_teaser|construction|specific_plan
        """
        ac.read_int()
        a = ac.read_list_ints()
        m = 2 * 10 ** 5
        cnt = [0] * (m + 1)
        for num in a:
            cnt[num] += 1
        ans = 0
        res = [-1, -1]
        pre = c = 0
        for num in range(1, m + 1):
            if cnt[num]:
                cur = c + cnt[num]
                if cur > ans:
                    ans = cur
                    res = [num - pre, num]
                if cnt[num] > 1:
                    pre += 1
                    c += cnt[num]
                else:
                    pre = 1
                    c = 1
            else:
                pre = c = 0
        ac.st(ans)
        lst = [res[0]]
        cnt[res[0]] -= 1
        for i in range(res[0] + 1, res[1] + 1):
            lst.append(i)
            cnt[i] -= 1
        for i in range(res[1], res[0] - 1, -1):
            lst.extend([i] * cnt[i])
        ac.lst(lst)
        return

    @staticmethod
    def cf_1183f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1183/problem/F
        tag: greedy|brain_teaser|classical|brute_force|special_judge
        """
        for _ in range(ac.read_int()):
            ac.read_int()
            nums = ac.read_list_ints()
            ceil = max(nums)
            ans = ceil
            for num in nums:
                if ceil % num:
                    if num + ceil > ans:
                        ans = num + ceil
            if ceil % 30 == 0 and ceil // 2 in nums and ceil // 3 in nums and ceil // 5 in nums:
                x, y, z = ceil // 2, ceil // 3, ceil // 5
                if x % y and y % z and x % z:
                    cur = x + y + z
                    if cur > ans:
                        ans = cur

            nums = [num for num in nums if ceil % num]
            if nums:
                ceil2 = max(nums)
                for num in nums:
                    if ceil2 % num:
                        if num + ceil2 + ceil > ans:
                            ans = num + ceil2 + ceil
            ac.st(ans)

        return

    @staticmethod
    def cf_1883f(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1883/problem/F
        tag: brain_teaser|prefix_suffix
        """
        ac.get_random_seed()
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()

            last = dict()
            for i in range(n):
                last[nums[i] ^ ac.random_seed] = i

            first = dict()
            for i in range(n - 1, -1, -1):
                first[nums[i] ^ ac.random_seed] = i

            ans = pre = 0
            for i in range(n):
                if first[nums[i] ^ ac.random_seed] == i:
                    pre += 1
                if last[nums[i] ^ ac.random_seed] == i:
                    ans += pre

            ac.st(ans)
        return

    @staticmethod
    def abc_293f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc293/tasks/abc293_f
        tag: binary_search|brute_force|brain_teaser|classical
        """
        assert 1 << 64 > 10 ** 18

        def compute(bb):
            res = 0
            for w in cur:
                res = res * bb + w
            return res

        def check(bb):
            return compute(bb) >= n

        for _ in range(ac.read_int()):
            n = ac.read_int()
            ans = {n, n - 1}

            for num in range(2, 1 << 5):
                cur = [int(w) for w in bin(num)[2:]]
                x = BinarySearch().find_int_left(0, n, check)
                if compute(x) == n:
                    ans.add(x)

            for x in range(5, 65):
                if 1 << x > n:
                    break
                high = int(n ** (1 / x))
                low = int(n ** (1 / (x + 1)))
                for b in range(max(low, 2), min(high + 10, n - 1)):
                    num = n
                    while num:
                        if num % b > 1:
                            break
                        num //= b
                    else:
                        ans.add(b)

            ac.st(len([x for x in ans if x >= 2]))
        return

    @staticmethod
    def abc_252f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc252/tasks/abc252_f
        tag: greedy|small_to_big|reverse_order|classical
        """
        n, ll = ac.read_list_ints()
        nums = ac.read_list_ints()
        tot = sum(nums)
        if ll > tot:
            nums.append(ll - tot)
        heapify(nums)
        ans = 0
        while len(nums) >= 2:
            a, b = heappop(nums), heappop(nums)
            ans += a + b
            heappush(nums, a + b)
        ac.st(ans)
        return

    @staticmethod
    def abc_349d(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc349/tasks/abc349_d
        tag: greedy|brain_teaser
        """
        lst = []
        for _ in range(3):
            lst.extend(ac.read_list_ints())
        ind = list()
        ind.append([[i, i] for i in range(3)])
        ind.append([[i, 2 - i] for i in range(3)])
        ind.extend([[i, j] for j in range(3)] for i in range(3))
        ind.extend([[i, j] for i in range(3)] for j in range(3))
        ll, rr = ac.read_list_ints()
        ans = []
        while ll < rr:
            for i in range(60, -1, -1):
                if ll % (1 << i) == 0 and ll + (1 << i) <= rr:
                    ans.append((ll, ll + (1 << i)))
                    ll += 1 << i
                    break
        ac.st(len(ans))
        for a in ans:
            ac.lst(a)
        return

    @staticmethod
    def abc_249f(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc249/tasks/abc249_f
        tag: greedy|implemention|reverse_order|classical
        """
        n, k = ac.read_list_ints()
        not_use = []
        post = 0
        nums = [[1, 0]] + [ac.read_list_ints() for _ in range(n)]
        nums.reverse()
        ans = -math.inf
        for t, y in nums:
            if t == 1:
                while len(not_use) > k:
                    x = heapq.heappop(not_use)
                    post -= x
                k -= 1
                ans = max(ans, post + y)
                if k < 0:
                    break
            else:
                if y >= 0:
                    post += y
                else:
                    heapq.heappush(not_use, -y)
        ac.st(ans)
        return

    @staticmethod
    def abc_229g(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc229/tasks/abc229_g
        tag: implemention|median_greedy|two_pointers|classical|prefix_sum
        """
        s = ac.read_str()
        k = ac.read_int()
        n = len(s)
        ind = [i for i in range(n) if s[i] == "Y"]
        m = len(ind)
        pre = ac.accumulate(ind)

        def check(x, y):
            mid = (x + y) // 2
            res = 0
            cnt = mid - x
            if cnt:
                start = ind[mid] - cnt
                res += (start + start + cnt - 1) * cnt // 2 - (pre[mid] - pre[x])

            cnt = y - mid
            if cnt:
                start = ind[mid] + 1
                res += -(start + start + cnt - 1) * cnt // 2 + (pre[y + 1] - pre[mid + 1])
            return res

        ans = j = 0
        for i in range(m):
            if j < i:
                j = i
            while j + 1 < m and check(i, j + 1) <= k:
                j += 1
            ans = max(ans, j - i + 1)
        ac.st(ans)
        return

    @staticmethod
    def cf_1974g(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1974/problem/G
        tag: regret_heapq|implemention|brain_teaser|classical
        """
        for _ in range(ac.read_int()):
            m, x = ac.read_list_ints()
            nums = ac.read_list_ints()
            stack = []
            ans = 0
            pre = x
            for num in nums[1:]:
                if pre >= num:
                    ans += 1
                    pre -= num
                    heapq.heappush(stack, -num)
                elif stack and -stack[0] > num:
                    pre -= heapq.heappop(stack)
                    ans -= 1
                    if pre >= num:
                        ans += 1
                        pre -= num
                        heapq.heappush(stack, -num)
                pre += x
            ac.st(ans)
        return

    @staticmethod
    def cf_1976b(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1976/problem/B
        tag: brute_force|greedy
        """
        for _ in range(ac.read_int()):
            ac.get_random_seed()
            n = ac.read_int()
            a = ac.read_list_ints()
            b = ac.read_list_ints()
            tot = sum(abs(a[i] - b[i]) for i in range(n))
            ans = math.inf
            target = b[-1]
            for i in range(n):
                num = a[i]
                cur = tot - abs(a[i] - b[i])
                bb = b[i]
                if num <= target <= b[i] or num >= target >= b[i]:
                    cur += 1 + abs(a[i] - b[i])
                else:
                    if target <= num <= bb or bb <= num <= target:
                        cur += abs(num - target) + 1 + abs(a[i] - b[i])
                    else:
                        cur += abs(bb - target) + 1 + abs(a[i] - b[i])
                ans = min(ans, cur)
            ac.st(ans)
        return

    @staticmethod
    def abc_209c(ac=FastIO()):
        """
        url: https://atcoder.jp/contests/abc209/tasks/abc209_c
        tag: greedy|brain_teaser
        """
        ac.read_int()
        nums = ac.read_list_ints()
        nums.sort()
        mod = 10 ** 9 + 7
        ans = 1
        for i, num in enumerate(nums):
            if num <= i:
                ans = 0
                break
            ans *= (num - i)
            ans %= mod
        ac.st(ans)
        return

    @staticmethod
    def cf_985c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/985/C
        tag: greedy|brain_teaser|reverse_order
        """
        n, k, ll = ac.read_list_ints()
        nums = ac.read_list_ints()
        nums.sort()
        post = ans = 0
        ceil = nums[0] + ll
        for i in range(n * k - 1, -1, -1):
            if nums[i] <= ceil and post >= k - 1:
                post -= k - 1
                ans += nums[i]
            else:
                post += 1
        ac.st(ans if post == 0 else 0)
        return

    @staticmethod
    def cf_1156c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1156/C
        tag: greedy|two_pointers|classical|brain_teaser
        """
        n, z = ac.read_list_ints()
        nums = ac.read_list_ints()
        nums.sort()
        ans = 0
        j = n//2
        for i in range(n//2):
            while j < n and nums[j] < z + nums[i]:
                j += 1
            if j < n:
                ans += 1
                j += 1
            else:
                break
        ac.st(ans)
        return

    @staticmethod
    def cf_1684d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1684/D
        tag: greedy|observation|contribution_method
        """
        for _ in range(ac.read_int()):
            n, k = ac.read_list_ints()
            nums = ac.read_list_ints()
            ind = list(range(n))
            ind.sort(key=lambda it: nums[it] - (n - it - 1), reverse=True)
            for i in ind[:k]:
                nums[i] = 0
            ans = pre = 0
            for i in range(n):
                if nums[i] == 0:
                    pre += 1
                else:
                    ans += nums[i] + pre
            ac.st(ans)
        return

    @staticmethod
    def cf_1379c(ac=FastIO()):
        """
        url: https://codeforces.com/contest/1379/problem/C
        tag: observation|prefix_sum|binary_search|brute_force|greedy
        """
        q = ac.read_int()
        for i in range(q):
            n, m = ac.read_list_ints()
            nums = [ac.read_list_ints() for _ in range(m)]
            if i < q - 1:
                ac.read_str()
            lst = [a for a, _ in nums]
            lst.sort()
            if n == 1:
                ac.st(max(lst))
                continue

            pre = ac.accumulate(lst)
            ans = max(lst[-n:])
            for a, b in nums:
                rest = n - 2
                cur = a + b
                i = bisect.bisect_left(lst, b)
                x = min(rest, m - i)
                cur += pre[m] - pre[m - x]
                if x and a > b:
                    x -= 1
                    cur -= a
                rest -= x
                cur += rest * b
                ans = max(ans, cur)
            ac.st(ans)
        return

    @staticmethod
    def cf_1415d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1415/D
        tag: observation|bit_operation|data_range
        """
        n = ac.read_int()
        nums = ac.read_list_ints()
        if n > 65:
            ac.st(1)
            return
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] ^ nums[i]

        ans = math.inf
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(i, j):
                    if pre[k + 1] ^ pre[i] > pre[j + 1] ^ pre[k + 1]:
                        ans = min(ans, j - i - 1)
        ac.st(ans if ans < math.inf else -1)
        return

    @staticmethod
    def lg_p2968(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P2968
        tag: greedy|implemention|observation
        """
        pos = 0
        speed = 1
        ans = 1
        n, k = ac.read_list_ints()
        nums = [ac.read_list_ints() for _ in range(k)]
        nums.sort()
        for i in range(k - 2, -1, -1):
            nums[i][1] = min(nums[i][1], nums[i + 1][1] + (nums[i + 1][0] - nums[i][0]))
        for p, s in nums:
            x = min((p + s - pos - speed) // 2, p - pos)
            ans = max(ans, speed + x)
            speed = min(speed + x, s)
            pos = p
        ans = max(ans, speed + n - pos)
        ac.st(ans)
        return

    @staticmethod
    def lg_p4823(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P4823
        tag: greedy|regret_heapq|classical|brain_teaser
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        h = ac.read_int()
        nums.sort(key=lambda x: x[0] + x[1])
        post = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            post[i] = post[i + 1] + nums[i][0]
        ans = tot = 0
        stack = []
        for i, (a, b) in enumerate(nums):
            if tot + post[i] + b >= h:
                heappush(stack, -a)
                ans += 1
            else:
                if stack and -stack[0] > a:
                    tot -= heappop(stack)
                    heappush(stack, -a)
                else:
                    tot += a
        ac.st(ans)
        return

    @staticmethod
    def lg_p6002(ac=FastIO()):
        """
        url: https://www.luogu.com.cn/problem/P6002
        tag: brute_force|greedy|brain_teaser
        """
        n, k = ac.read_list_ints()
        nums = ac.read_list_ints()
        ans = 0
        for x in range(1, 1001):
            cur = sum(num // x for num in nums)
            if cur >= k:
                ans = max(ans, x * k // 2)
            elif cur >= k // 2:
                lst = sorted([num % x for num in nums], reverse=True)
                ans = max(ans, (cur - k // 2) * x + sum(lst[:k - cur]))
            else:
                break
        ac.st(ans)
        return

    @staticmethod
    def cf_1257d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1257/D
        tag: suffix_max|greedy|implemention|classical
        """
        for _ in range(ac.read_int()):
            n = ac.read_int()
            nums = ac.read_list_ints()
            m = ac.read_int()
            post = [0] * (n + 1)
            for _ in range(m):
                p, s = ac.read_list_ints()
                post[s] = max(post[s], p)
            for i in range(n - 1, -1, -1):
                post[i] = max(post[i], post[i + 1])
            if max(nums) > post[1]:
                ac.st(-1)
                continue
            ans = i = 0
            while i < n:
                j = i
                cur = nums[i]
                while j + 1 < n and max(cur, nums[j + 1]) <= post[j - i + 2]:
                    cur = max(cur, nums[j + 1])
                    j += 1
                ans += 1
                i = j + 1
            ac.st(ans)
        return

    @staticmethod
    def cf_1539d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/1539/D
        tag: greedy|two_pointers|implemention
        """
        n = ac.read_int()
        nums = [ac.read_list_ints() for _ in range(n)]
        nums.sort(key=lambda it: -it[1])
        i, j = 0, n - 1
        ans = 0
        cost = 0
        while i <= j:
            if nums[j][0] == 0:
                j -= 1
                continue
            if nums[i][0] == 0:
                i += 1
                continue
            if ans >= nums[j][1]:
                ans += nums[j][0]
                cost += nums[j][0]
                nums[j][0] = 0
                j -= 1
                continue
            cur = min(nums[j][1] - ans, nums[i][0])
            ans += cur
            cost += cur * 2
            nums[i][0] -= cur

        ac.st(cost)
        return

    @staticmethod
    def cf_865d(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/865/D
        tag: regret_heapq|greedy|classical
        """
        ac.read_int()
        nums = ac.read_list_ints()
        ans = 0
        pre = []
        for num in nums:
            if pre and num > pre[0]:
                cur = heappop(pre)
                ans += num - cur
                heappush(pre, num)
            heappush(pre, num)
        ac.st(ans)
        return

    @staticmethod
    def cf_713c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/713/C
        tag: greedy|brain_teaser|strictly_monotonic_trick|classical
        """
        n = ac.read_int()
        ans = 0
        stack = []
        nums = ac.read_list_ints()
        for i in range(n):
            x = nums[i] - i  # important
            heappush(stack, -x)
            if stack and x < -stack[0]:
                heappush(stack, -x)
                ans += -heappop(stack) - x
        ac.st(ans)
        return

    @staticmethod
    def cf_13c(ac=FastIO()):
        """
        url: https://codeforces.com/problemset/problem/13/C
        tag: greedy|brain_teaser|regret_heapq|classical
        """
        n = ac.read_int()
        ans = 0
        stack = []
        nums = ac.read_list_ints()
        for i in range(n):
            x = nums[i]
            heappush(stack, -x)
            if stack and x < -stack[0]:
                heappush(stack, -x)
                ans += -heappop(stack) - x
        ac.st(ans)
        return
