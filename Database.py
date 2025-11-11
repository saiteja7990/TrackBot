import pymysql

h="localhost"
u="root"
p="root"
d="finance"

def getConnection():
    con = pymysql.connect(host=h,user=u,password=p,database=d)
    return con
