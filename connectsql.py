import mysql.connector as mc
import time
import datetime

connect = mc.connect(host= "localhost", user = 'root', password = '1234', port = 3307, database = 'traffic')
a = time.ctime()
t = a.split(" ")[4]
min = t.split(":")[1]
hour = int(t.split(":")[0])
date = str(datetime.date.today())
big, mid, small = 45, 121, 98
if connect.is_connected():
    cursor = connect.cursor()
    cursor.execute(f"Insert into format values('{date}', {hour}, {big}, {mid}, {small})")
    connect.commit()
    print(f'query successfully executed at {a}')
connect.close()