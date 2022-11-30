import json
import requests

# 力扣目前勋章的配置
RATING = 1600
GUARDIAN = 0.05
KNIGHT = 0.25
# 查询的地址为全国还是全球？很关键
GLOBAL = False
# 二分查找的右端点(可自调)
RIGHT = 4000


class RankingCrawler:
    URL = 'https://leetcode.com/graphql' if GLOBAL else 'https://leetcode-cn.com/graphql'
    URL = 'https://leetcode.cn/contest/weekly-contest-307/ranking/2/'
    _REQUEST_PAYLOAD_TEMPLATE = {
        "operationName": None,
        "variables": {},
        "query":
            r'''{
                localRanking(page: 1) {
                    totalUsers
                    userPerPage
                    rankingNodes {
                        currentRating
                        currentGlobalRanking
                    }
                }
            }
            ''' if not GLOBAL else
            r'''{
                globalRanking(page: 1) {
                    totalUsers
                    userPerPage
                    rankingNodes {
                        currentRating
                        currentGlobalRanking
                    }
                }
            }
            '''
    }

    def fetch_lastest_ranking(self, mode):
        l, r = 1, RIGHT
        retry_cnt = 0
        ansRanking = None
        while l < r:
            cur_page = (l + r + 1) // 2
            try:
                payload = RankingCrawler._REQUEST_PAYLOAD_TEMPLATE.copy()
                payload['query'] = payload['query'].replace('page: 1', 'page: {}'.format(cur_page))
                resp = requests.post(RankingCrawler.URL,
                                     headers = {'Content-type': 'application/json'},
                                     json = payload).json()

                resp = resp['data']['localRanking'] if not GLOBAL else resp['data']['globalRanking']
                # no more data
                if len(resp['rankingNodes']) == 0:
                    break
                if not mode:
                    top = int(resp['rankingNodes'][0]['currentRating'].split('.')[0])
                    if top < RATING:
                        r = cur_page - 1
                    else:
                        l = cur_page
                        ansRanking = resp['rankingNodes']
                else:
                    top = int(resp['rankingNodes'][0]['currentGlobalRanking'])
                    if top > mode:
                        r = cur_page - 1
                    else:
                        l = cur_page
                        ansRanking = resp['rankingNodes']

                print('The first contest current rating in page {} is {} .'.format(cur_page, resp['rankingNodes'][0]['currentRating']))
                retry_cnt = 0
            except:
                # print(f'Failed to retrieved data of page {cur_page}...retry...{retry_cnt}')
                retry_cnt += 1
        ansRanking = ansRanking[::-1]
        last = None
        if not mode:
            while ansRanking and int(ansRanking[-1]['currentRating'].split('.')[0]) >= 1600:
                last = ansRanking.pop()
        else:
            while ansRanking and int(ansRanking[-1]['currentGlobalRanking']) <= mode:
                last = ansRanking.pop()
        return last


if __name__ == "__main__":
    print('start')
    crawler = RankingCrawler()
    ans = crawler.fetch_lastest_ranking(0)
    n = int(ans['currentGlobalRanking'])
    guardian = crawler.fetch_lastest_ranking(int(GUARDIAN * n))
    knight = crawler.fetch_lastest_ranking(int(KNIGHT * n))
    if not GLOBAL:
        guardian['currentCNRanking'] = guardian['currentGlobalRanking']
        guardian.pop('currentGlobalRanking')
        knight['currentCNRanking'] = knight['currentGlobalRanking']
        knight.pop('currentGlobalRanking')

    print("Done!")
    print()
    print("目前全{}1600分以上的有{}人".format("球" if GLOBAL else "国",n))
    print("根据这个人数，我们得到的Guardian排名及分数信息为:{}".format(guardian))
    print("根据这个人数，我们得到的Knight排名及分数信息为:{}".format(knight))

