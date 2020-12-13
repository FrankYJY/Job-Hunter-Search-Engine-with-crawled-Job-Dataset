import requests,json,csv,time

# 获取数据
# get information
session = requests.session()

# 模拟浏览器向网站发起请求
# Sims Browsers Initiate Requests for Websites
headers = {
    'referer' : 'https://www.lagou.com/jobs/list_%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90?&px=default&city=%E5%85%A8%E5%9B%B',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'
}
session.get('https://www.lagou.com/jobs/list_%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90/p-city_0?px=default',headers=headers)

# 模拟拉勾网请求内部数据
# Simulated pull-up network request internal data
url = 'https://www.lagou.com/jobs/positionAjax.json?px=default&needAddtionalResult=false'

kd = input("请输入你想查找的岗位：")
for i in range(1,30):
    #便于查看结果写了print
    print(f'开始爬取第{i}页')
    data = {
        'first': 'false',
        'pn': str(i),
        'kd': kd,
        'sid': '3a604e19a96a40669b9b84b799726000'
    }
    reponse = session.post(url=url, headers=headers, data=data, cookies=session.cookies)
    text = json.loads(reponse.text)
    data = text.get('content').get('positionResult').get('result')

    # 解析数据
    # analyze data
    for i in data:
        info = []
        #添加各类信息
        info.append(i['positionName'])
        info.append(i['companyId'])
        info.append(i['companyShortName'])
        info.append(i['companySize'])
        info.append(i['industryField'])
        info.append(i['workYear'])
        info.append(i['city'])
        info.append(i['salary'])
        info.append(i['education'])
        info.append(i['jobNature'])
        info.append(i['companyLabelList'])
        info.append(i['positionAdvantage'])
        print(info)

        # 保存数据至本地
        # store
        with open('lagou_ITjobs.csv', 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(info)

        # 睡眠1秒
        # set interval
        time.sleep(1)

