import psycopg2 as pg

host,port,user,password,database = open('dbinfo','r').read().split(',')
con = pg.connect(host=host, database=database, user=user, password=password, port=port)
cur = con.cursor()
table_names = {'JUROS':'risco_juros','INFLACAO':'risco_inflacao','CAMBIO':'risco_cambio','TRADING':'risco_trading','MSO':'risco_mso'}

cur.execute('SELECT * FROM etl.risco_cambio;')
print(cur.fetchall())