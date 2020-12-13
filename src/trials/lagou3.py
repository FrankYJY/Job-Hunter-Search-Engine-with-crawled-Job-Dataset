import requests
import json
import time

url_request = 'https://www.lagou.com/jobs/positionAjax.json?needAddtionalResult=false'   #network -> headers中显示的请求链接
url_html = 'https://www.lagou.com/jobs/list_python?labelWords=&fromSearch=true&suginput='   #网页显示访问链接
hd = {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Referer': 'https://www.lagou.com/jobs/list_python?labelWords=&fromSearch=true&suginput=',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'
}

'''
爬取多页职位信息

for i in range(1,6):
    data = {
        'first': 'true',
        'pn': 'str(i)',
        'kd': 'python'
}
'''

#根据network -> headers显示的data来设置data
data = {
        'first': 'true',
        'pn': '1',
        'kd': 'python'
}

# 创建一个session对象
s = requests.Session()
#发送请求，获得cookies
s.get(url_html,headers = hd,data = data,timeout = 4)
cookie = s.cookies
response = s.post(url_request,data = data,headers = hd,cookies = cookie,timeout = 4) #获得此次文本
#print(response)
response.encoding = response.apparent_encoding
time.sleep(6)
#response.encoding = response.apparent_encoding
#print(response.text)
text = json.loads(response.text)
#print(text)
info = text["content"]["positionResult"]["result"]
for i in info:
    #print(i)
    print(i["companyFullName"])
    companyFullName = i["companyFullName"]
    print(i["positionName"])
    positionName = i["positionName"]
    print(i["salary"])
    salary = i["salary"]
    print(i["companySize"])
    companySize = i["companySize"]
    print(i["skillLables"])
    skillLables = i["skillLables"]
    print(i["createTime"])
    createTime = i["createTime"]
    print(i["district"])
    district = i["district"]
    print(i["stationname"])
    stationname = i["stationname"]
    print("-------------------------")
