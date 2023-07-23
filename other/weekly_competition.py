from requests_html import HTMLSession
import json
import requests

import urllib
import pandas as pd
def get_rank_info():
    url='https://leetcode.cn/contest/weekly-contest-307/ranking/2/'

    #url="http://data.sports.sohu.com/nba/nba_teams_rank.php?type=division#division"
    tables = pd.read_html(url)
    print("table数量:",len(tables))
    print(tables)
    #
    # session = HTMLSession()
    # url = 'https://leetcode.cn/contest/weekly-contest-307/ranking/2/'
    # r = session.get(url)
    # print(r.html.text)
    # #resp = requests.post(url, headers = {'Content-type': 'application/json'})
    # #print(resp)

if __name__ == "__main__":
    get_rank_info()
