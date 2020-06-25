##GitHub爬虫
单独爬虫而言，使用*spider_3_2*分支即可

爬虫的启动文件在  *source/data/service/ProjectAllDataFetcher.py*

爬虫需要修改配置文件  *source/config/config.txt*

主要流程是从配置文件读取参数，向GitHub api发起请求，并存入数据库

###配置文件参数说明

* ***token***  API认证使用的Token，多个Token依次罗列，逗号分割   
* ***host username password database***   远程连接数据库配置
* ***print***  爬虫运行输出参数
* ***owner repo*** 指定项目的名称 示例:*rails/rails*
* ***limit start*** 项目获取pull-request编号范围 [start-limit, start]
* ***timeout*** 网络请求超时判定失败时间
* ***retry*** 单个网络请求最多重试次数
* ***proxy*** 是否使用代理ip池   使用时必须配合*proxy_pool*,否则False
* ***semaphore*** 异步同步信号量
* ***api*** 爬虫api选择，3为reset，4为graphql

###代理ip池说明
配置文件 ***proxy*** 字段True时候，必须本地运行开源项目*proxy_pool*使用，
proxy_pool 配置参考*https://github.com/jhao104/proxy_pool*
开启代理ip可以显著增加接口请求的成功率，但是也降低了Token被封的风险

切记不要用自己的真实账号的Token，被封了后果自负！不排除使用其他Token导致
某个设备上面登陆过的账号全部封禁的风险！！