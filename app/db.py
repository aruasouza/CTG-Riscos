import psycopg2 as pg
import pandas as pd

host,port,user,password,database = open('dbinfo','r').read().split(',')
con = pg.connect(host=host, database=database, user=user, password=password, port=port)
cur = con.cursor()
table_names = {'JUROS':'risco_juros','INFLACAO':'risco_inflacao','CAMBIO':'risco_cambio','TRADING':'risco_trading','MSO':'risco_mso'}

def fetc_risco(risco,df):
    try:
        table = table_names[risco]
        clean_table = f'''
        DROP TABLE IF EXISTS etl.{table};
        CREATE TABLE etl.{table} (
        id SERIAL NOT NULL PRIMARY KEY,
        date TEXT,
        worst TEXT,
        base TEXT,
        best TEXT,
        prob_worst TEXT,
        prob_base TEXT,
        prob_best TEXT,
        probabilidade_impacto_negativo TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
        '''
        cur.execute(clean_table)
        values = list(df.itertuples(index=True))
        for dados in values:
            data = str(dados[0]).replace('-','').split(' ')[0]
            sql = f'INSERT INTO etl.{table} (date,worst,base,best,prob_worst,prob_base,prob_best,probabilidade_impacto_negativo) VALUES ({data},{dados[1]},{dados[2]},{dados[3]},{dados[4]},{dados[5]},{dados[6]},{dados[7]});'
            cur.execute(sql)
        con.commit()
    except Exception as e:
        print(e)