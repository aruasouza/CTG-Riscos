from setup import adlsFileSystemClient,multithread,ctg_temp_path,downloads_path,file_names
import pandas as pd
import os
import datetime
import montecarlo
import numpy as np
import json
from bcb import sgs
from models import error
import db
import requests
from multiprocessing.dummy import Pool
url_vini = 'https://api.ctgriscoscompliance.ianclive.com/etl'

def on_success(r):
    print(f'Request succeed: {r}')

def on_error(ex: Exception):
    print(f'Request failed: {ex}')

def upload_file(risco,tipo):
    file_name = f'{tipo}_{risco}.csv'
    file_path = os.path.join(ctg_temp_path,file_name)
    directory = f'DataLakeRiscoECompliance/RISCOS'
    multithread.ADLUploader(adlsFileSystemClient, lpath=file_path,
        rpath=f'{directory}/{risco}/{file_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)

def composed_interest(days,values):
    results = [values[0] + 1]
    for day,valor in zip(days[1:],values[1:]):
        if day == 0:
            results.append(1)
        else:
            results.append(results[-1] * valor + 1)
    return np.array(results) - 1

def absolute(serie,base):
    valor_atual = base
    yield valor_atual
    for valor in serie[:-1]:
        valor = valor / 100
        valor_atual += (valor_atual * valor)
        yield valor_atual

def dif_percent(column):
    lista = list(column.diff())[1:] + [None]
    return pd.Series(lista,index = column.index) / column

def ultimo_reajuste(row):
    mes_de_reajuste = row['ReajusteDataBase'].month
    mes_atual = row['Competencia'].month
    ano_atual = row['Competencia'].year
    if mes_atual >= mes_de_reajuste:
        data = pd.Period(f'{ano_atual}-{mes_de_reajuste}')
    else:
        data = pd.Period(f'{ano_atual - 1}-{mes_de_reajuste}')
    return max(data,row['ReajusteDataBase'])

def read_mso():
    mapa = {'Janeiro':1,'Fevereiro':2,'Março':3,'Abril':4,'Maio':5,'Junho':6,'Julho':7,'Agosto':8,'Setembro':9,'Outubro':10,'Novembro':11,'Dezembro':12}
    def to_date(data):
        mes,ano = data.split('/')
        return datetime.datetime(int(ano) + 2000,mapa[mes],1)
    filename = file_names['calculoinflacao']
    real = pd.read_excel(os.path.join(ctg_temp_path,filename),sheet_name = 'Por Mês',header = 7).iloc[:6].drop(['Unnamed: 0','Unnamed: 1'],axis = 1).T.dropna().sum(axis = 1)
    inf = pd.read_excel(os.path.join(ctg_temp_path,filename),sheet_name = 'Por Mês',header = 14).iloc[:6].drop(['Unnamed: 0','Unnamed: 1'],axis = 1).T.dropna().set_index(real.index).sum(axis = 1)
    deflator = pd.read_excel(os.path.join(ctg_temp_path,filename),sheet_name = 'Por Mês',header = 7).iloc[0,64]
    df = pd.DataFrame({'real':real.values,'calc':inf.values},index = list(map(to_date,real.index)))
    ano_corrente = datetime.datetime.now().year
    df.loc[:datetime.datetime(ano_corrente,12,31),'real'] = df.loc[:datetime.datetime(ano_corrente,12,31),'real'] / (deflator + 1)
    return df

def read_cash_dcf():
    filename = file_names['dcf-rp']
    dcf = pd.read_excel(os.path.join(ctg_temp_path,filename),sheet_name = 'Budget',header = 5)
    dcf = dcf.loc[~dcf['Unnamed: 1'].isnull()].drop('Unnamed: 0',axis = 1).set_index('Unnamed: 1').T
    dcf.columns.name = None
    dcf = dcf[list(map(lambda x: type(x) == datetime.datetime,dcf.index.values))]
    serie_selic = dcf[['Cash Accumulated Available']].rename({'Cash Accumulated Available':'Cash'},axis = 1) * 1000
    serie_selic = serie_selic.set_index(pd.DatetimeIndex(serie_selic.index))
    arquivo = os.path.join(ctg_temp_path,file_names['debt-rp'])
    dfs = []
    messages = []
    for i in range(1,11):
        try:
            df = pd.read_excel(arquivo,sheet_name = f'{i}ª Emissão CDI',header = 4)
            df = df[df['Filtro'] == 'X'][['Data','Last  CDI','Days','Interest (Month)','Principal Balance']].dropna()
            df['Data'] = pd.to_datetime(df['Data'],format = '%d%m%Y')
            if df['Data'].iloc[0] < datetime.datetime.now():
                df['Last  CDI'] = df['Last  CDI'] / 100
                spread = pd.read_excel(arquivo,sheet_name = f'{i}ª Emissão CDI').iloc[2,9] / 100
                df['Spread'] = spread
                df['Taxa Diária Estimada'] = ((df['Last  CDI'] + spread + 1) ** (1/252)) - 1
                df['Days Dif'] = df['Days'].diff().fillna(df['Days'].iloc[0]).apply(lambda x: 0 if x < 2 else x)
                df['Taxa Estimada Período'] = ((1 + df['Taxa Diária Estimada']) ** df['Days Dif']) - 1
                df['Taxa Acumulada Estimada'] = composed_interest(df['Days Dif'].values,df['Taxa Estimada Período'].values)
                df['Juros Estimados'] = df['Principal Balance'] * df['Taxa Acumulada Estimada']
                df['Período'] = df['Data'].apply(lambda x: str(x.year) + '-' + str(x.month) if len(str(x.month)) == 2 else str(x.year) + '-0' + str(x.month))
                dfs.append(df)
                messages.append(f'{i}ª Emissão CDI: Sucesso')
            else:
                messages.append(f'{i}ª Emissão CDI: Não Realizado')
        except ValueError as e:
            print(e)
            messages.append(f'{i}ª Emissão CDI: ' + str(e))
    return dfs,serie_selic[datetime.datetime.today():].copy()

def read_rp():
    filename = file_names['debt-rp']
    rp = pd.read_excel(os.path.join(ctg_temp_path,filename),sheet_name = 'Daily Calculation prop2019',header = 2)
    rp = rp[['Date','USD','Repayment']].dropna()
    rp = rp.set_index('Date')
    rp = rp[datetime.datetime.today():]
    rp['Period'] = rp.index.to_period('m')
    rp['Period'] = rp['Period'].apply(str)
    return rp

def read_deb_ipca():
    filename = file_names['debt-rp']
    arquivo = os.path.join(ctg_temp_path,filename)
    # positions = (15,14)
    dfs = []
    for i in range(9):
        try:
            emissao = i + 1
            df = pd.read_excel(arquivo,sheet_name = f'{emissao}ª Emissão IPCA',header = 6)
            df = df[df['Filtro'] == 'X'][['Date','IPCA','Interest','Monetary Variation',' Principal Balance']]
            df['Date'] = pd.to_datetime(df['Date'],format = '%d%m%Y')
            raw = pd.read_excel(arquivo,sheet_name = f'{emissao}ª Emissão IPCA')
            total = raw.iloc[0,15]
            juros = raw.iloc[4,15]
            del(raw)
            df['Taxa'] = juros
            df['Capital'] = total
            df['Days Dif'] = df['Date'].diff().apply(lambda x: x.days).fillna(30)
            df = df[df['Days Dif'] <= 1]
            df['Monetary Variation'] = df['Monetary Variation'].fillna(0)
            subtract = 0
            for i in df.index:
                if df.loc[i,'Days Dif'] == 1:
                    subtract += df.loc[i,'Interest']
                df.loc[i,'Capital'] -= subtract
            df = df[df['Days Dif'] != 1]
            df['Capital Dif'] = df['Capital'].diff().fillna(0).cumsum()
            df['Período'] = df['Date'].apply(lambda x: str(x.year) + '-' + str(x.month) if len(str(x.month)) == 2 else str(x.year) + '-0' + str(x.month))
            dfs.append(df)
        except ValueError as e:
            print(e)

    # costs = pd.read_excel(os.path.join(ctg_temp_path,"DCF - RP'22 8&04 - Valores.xlsx"),sheet_name = 'Budget',header = 5)
    # costs = costs.loc[~costs['Unnamed: 1'].isnull()].drop('Unnamed: 0',axis = 1).set_index('Unnamed: 1').T
    # costs.columns.name = None
    # costs = costs[list(map(lambda x: type(x) == datetime.datetime,costs.index.values))]
    # costs = pd.Series(costs['Operational Costs'].values * 100,index = list(map(lambda x: f'{x.year}-{x.month}',costs.index)))

    return dfs #,costs

def read_contracts():
    print('Obtendo contratos...')
    adl = adlsFileSystemClient
    files_list = list(filter(lambda x: (x.split('/')[-1][-8:-5] == 'wbc') and (x.split('/')[-1][-4:] == 'json'),adl.walk('LandingData/Comercial/wbc')))
    files_list.sort()
    name = files_list[-1]
    with adl.open(name, 'rb') as f:
        js = json.load(f)
    df = pd.DataFrame.from_records(js['Values'])
    del(js)
    df = df[['ReajusteDataBase','Competencia','QuantSolicitada','Valor','ValorReajustado','Movimentacao']].dropna()
    df['ReajusteDataBase'],df['Competencia'] = pd.to_datetime(df['ReajusteDataBase']).apply(lambda x: pd.Period(x,'M')),pd.to_datetime(df['Competencia']).apply(lambda x: pd.Period(x,'M'))
    df['UltimoReajuste'] = df.apply(ultimo_reajuste,axis = 1)
    df['Valor'] = df['Valor'].apply(float)
    df.loc[df['Movimentacao'] == 'Compra','Valor'] *= -1
    df.loc[df['Movimentacao'] == 'Compra','ValorReajustado'] *= -1
    df['Total'] = df['QuantSolicitada'] * df['ValorReajustado']
    ipca = sgs.get({'ipca':433},start = '2000-01-01')
    ipca['ipca'] = [valor for valor in absolute(ipca['ipca'].values,1598.41)]
    ipca = ipca.set_index(ipca.index.to_period('M'))
    df = df.join(ipca,on = 'UltimoReajuste')
    df = df.join(ipca,on = 'ReajusteDataBase',rsuffix = '_base')

    # Parte LCA
    filename = file_names['lca']
    lca_anual = pd.read_excel(os.path.join(ctg_temp_path,filename),'Base_Anual',header = 11).dropna().set_index('Unnamed: 0').T['IPCA - IBGE (% a.a.)']
    lca_anual = pd.Series(lca_anual.values,index = list(map(lambda x: int(str(x)[:4]),lca_anual.index))).loc[2000:]
    lca_anual = pd.Series([valor for valor in absolute(lca_anual.values,1598.41)],index = lca_anual.index,name = 'ipca')
    df_anual = pd.DataFrame(index = pd.period_range(start = '{}-01-01'.format(lca_anual.index[0]),end = '{}-01-01'.format(lca_anual.index[-1]),freq = 'M'))
    df_anual['ano'] = df_anual.index.year
    df_anual = df_anual.join(lca_anual,on = 'ano')
    df_anual['lca_anual'] = list(df_anual['ipca'].rolling(12).mean())[12:] + ([None] * 12)
    lca_anual = df_anual['lca_anual']
    lca_mensal = pd.read_excel(os.path.join(ctg_temp_path,filename),'Base_Mensal',header = 8).drop([0,1,2]).set_index('Período')['IPCA']
    first_year = lca_mensal.index.year[0]
    lca_mensal = pd.Series([valor for valor in absolute(lca_mensal.values,lca_anual[f'{first_year}-01'])],index = lca_mensal.index.to_period('M'),name = 'lca_mensal')
    df = df.join(lca_mensal,on = 'UltimoReajuste')
    df = df.join(lca_mensal,on = 'ReajusteDataBase',rsuffix = '_base')
    df = df.join(lca_anual,on = 'UltimoReajuste')
    df = df.join(lca_anual,on = 'ReajusteDataBase',rsuffix = '_base')
    df['lca_mensal'] = df['lca_mensal'].fillna(df['lca_anual'])
    df['lca_mensal_base'] = df['lca_mensal_base'].fillna(df['lca_anual_base'])
    del(df['lca_anual'])
    del(df['lca_anual_base'])
    df['ValorReajustado'] = (df['lca_mensal'] / df['lca_mensal_base']) * df['Valor']
    df['Total'] = df['QuantSolicitada'] * df['ValorReajustado']

    return df.sort_values('Competencia')

def retrieve_forecast(risco):
    montecarlo.find_files(risco)
    try:
        montecarlo.find_datas(-1)
        return montecarlo.main_dataframe
    except Exception as e:
        print(e)

def shuffle(arr):
    new = arr.copy()
    np.random.shuffle(new)
    return new

def sum_series(*args):
    return pd.concat(args,axis = 1).sum(axis = 1)

def risco_juros(selic,dcf):
    size = 10000
    std = selic['std'].iloc[0]
    selic['Período'] = selic['date']
    date_range = pd.to_datetime(selic['date'],format = '%Y-%m')
    selic = selic.set_index('Período')
    cen_df = pd.concat([(selic['prediction'].rename(i) + pd.Series(np.random.normal(scale = std,size = len(selic)),index = selic.index).cumsum()) for i in range(size)],axis = 1)
    first_date = date_range[0]
    dfs = dcf[0]
    budg = dcf[1]

    # Caixa
    cen_base = pd.Series([None] * len(cen_df),cen_df.index)
    for df in dfs:
        temp = df.copy()
        temp = temp.drop_duplicates('Período').set_index('Período')
        cen_base = cen_base.fillna(temp['Last  CDI'])
    cen_base = ((1 + cen_base) ** (1/12)) - 1
    budg = budg[first_date:].copy()
    budg['Date'] = pd.Series(budg.index).apply(lambda x: str(x.year) + '-' + str(x.month) if len(str(x.month)) == 2 else str(x.year) + '-0' + str(x.month)).values
    budg = budg.set_index('Date')
    cen_caixa = pd.concat([(budg['Cash'].rename(i) * (cen_df[i] / 100)) - (budg['Cash'].rename(i) * cen_base) for i in cen_df.columns],axis = 1).fillna(0).cumsum().set_index(date_range)

    # Debentures
    for i,df in enumerate(dfs):
        df = df[df['Data'] >= first_date].copy()
        df['Despesa Estimada'] = df['Juros Estimados'].cumsum()
        df['Despesa Real'] = (df['Interest (Month)'] * df['Days Dif'].apply(lambda x: 0 if x == 0 else 1)).cumsum()
        df['Erro'] = (df['Despesa Real'] - df['Despesa Estimada']) / df['Despesa Real']
        dfs[i] = df
    cenarios = []
    for col in cen_df.columns:
        cenario = cen_df[[col]] / 100
        parciais = []
        for df in dfs:
            temp = df.join(cenario,on = 'Período')
            temp['Taxa Anual'] = ((temp[col] + 1) ** 12) - 1 + temp['Spread']
            temp['Taxa Diária Estimada'] = ((temp['Taxa Anual'] + 1) ** (1/252)) - 1
            temp['Taxa Estimada Período'] = ((1 + temp['Taxa Diária Estimada']) ** temp['Days Dif']) - 1
            temp['Taxa Acumulada Estimada'] = composed_interest(temp['Days Dif'].values,temp['Taxa Estimada Período'].values)
            temp['Juros Estimados'] = temp['Principal Balance'] * temp['Taxa Acumulada Estimada']
            temp['Despesa Estimada'] = temp['Juros Estimados'].cumsum()
            temp['Despesa Corrigida'] = temp['Despesa Estimada'] / (1 - (temp['Erro']))
            temp['Diferença Despesa'] = temp['Despesa Real'] - temp['Despesa Corrigida']
            temp = temp[temp['Days Dif'] == 0].drop_duplicates('Período')
            cenario_parcial = pd.Series(temp['Diferença Despesa'].values,index = temp['Período'].values)
            parciais.append(cenario_parcial)
        cenarios.append(sum_series(*parciais))
        if len(cenarios) % 100 == 0:
            print(len(cenarios))
    final = pd.concat(cenarios,axis = 1)
    final = pd.DataFrame(index = date_range).join(final.set_index(pd.to_datetime(final.index,format = '%Y-%m'))).fillna(method = 'ffill').fillna(0)
    return final + cen_caixa

def risco_ipca(ipca,dfs):
    # dfs,cost = files
    size = 10000
    std = ipca['std'].iloc[0]
    ipca['Período'] = ipca['date']
    date_range = pd.to_datetime(ipca['date'],format = '%Y-%m')
    first_date = date_range[0]
    ipca = ipca.set_index('Período')
    cen_df = (pd.concat([pd.Series(np.random.normal(scale = std,size = len(ipca)),index = ipca.index) for _ in range(size)],axis = 1).cumsum().T + ipca['prediction']).T
    
    # # despesas
    # cen_df_percent = cen_df.apply(dif_percent)
    # cen_df_percent['costs'] = cost
    # for cen in cen_df.columns:
    #     cen_df_percent[cen] = cen_df_percent['costs'] * (1 + cen_df_percent[cen])
    # cen_df_costs = cen_df_percent.drop('costs',axis = 1).fillna(0).cumsum().set_index(pd.to_datetime(cen_df.index,format = '%Y-%m'))
    
    # Debentures
    cenarios = []
    for col in cen_df.columns:
        cenario = cen_df[[col]]
        parciais = []
        for df in dfs:
            temp = df.join(cenario,on = 'Período')
            temp[col] = temp[col].fillna(temp['IPCA'])
            total = temp['Capital'].iloc[0]
            juros = temp['Taxa'].iloc[0]
            juros_semestral = ((1 + (juros / 100)) ** (1/2)) - 1
            ipca_base = (temp['Capital'].iloc[0] / temp[' Principal Balance'].iloc[0]) * temp[col].iloc[0]
            temp['Principal Balance Calculado'] = (total * temp[col] / ipca_base) + (temp['Capital Dif'] * temp[col] / ipca_base)
            temp['Interest Calculado'] = temp['Principal Balance Calculado'] * juros_semestral
            faltante = (total + sum(temp['Capital'].diff().dropna())) * (-1)
            temp['Monetary Variation Calculado'] = temp['Capital'].diff().apply(lambda x: None if x == 0 else x).fillna(method = 'bfill').fillna(faltante) * (-1) * ((temp['IPCA'] / ipca_base) - 1) * temp['Monetary Variation'].apply(lambda x: 1 if x > 0 else 0)
            temp = temp[temp['Date'] >= first_date]
            temp['Despesa Acumulada'] = (temp['Interest'] + temp['Monetary Variation']).cumsum()
            temp['Despesa Acumulada Calculado'] = (temp['Interest Calculado'] + temp['Monetary Variation Calculado']).cumsum()
            temp['Impacto'] = temp['Despesa Acumulada'] - temp['Despesa Acumulada Calculado']
            parciais.append(temp.set_index('Período')['Impacto'])
        cenarios.append(sum_series(*parciais))
        if len(cenarios) % 100 == 0:
            print(len(cenarios))
    final = pd.concat(cenarios,axis = 1)
    final_deb = pd.DataFrame(index = date_range).join(final.set_index(pd.to_datetime(final.index,format = '%Y-%m'))).fillna(method = 'ffill').fillna(0).replace(to_replace=0, method='ffill')
    return final_deb

def risco_cambio(cambio,rp):
    size = 10000
    risco = rp.copy()
    std = cambio['std'].iloc[0]
    cambio = cambio.join(risco.set_index('Period')[['USD','Repayment']],on = 'date').fillna(0)
    cen_df = ((pd.concat([pd.Series(np.random.normal(scale = std,size = len(cambio)),index = cambio.index) for _ in range(size)],axis = 1).cumsum().T + cambio['USD'] - cambio['prediction']) * cambio['Repayment']).T
    return cen_df.cumsum().set_index(pd.to_datetime(cambio['date']))

def risco_generico(df):
    size = 10000
    df['date'] = pd.to_datetime(df['date'],format = '%Y-%m')
    df = df.set_index('date')
    std = df['std'].iloc[0]
    simulation = np.random.normal(size = size,scale = std)
    cen_df = pd.concat([df['prediction'] + sim for sim in simulation],axis = 1)
    cen_df.columns = list(range(size))
    return cen_df.apply(lambda x: pd.Series(shuffle(x.values),index = x.index),axis = 1)

def risco_trading(ipca,con):
    size = 10000
    ipca['date'] = ipca['date'].apply(lambda x: pd.Period(x))
    ipca = ipca.set_index('date')
    std = ipca['std'].iloc[0]
    first_date = ipca.index[0]
    last_date = ipca.index[-1]
    con = con[(con['Competencia'] >= first_date) & (con['Competencia'] <= last_date)]
    cen_df = (pd.concat([pd.Series(np.random.normal(scale = std,size = len(ipca)),index = ipca.index) for _ in range(size)],axis = 1).cumsum().T + ipca['prediction']).T
    cenarios = []
    for i in cen_df.columns:
        cen = cen_df[[i]]
        temp = con.copy()
        temp = temp.join(cen,on = 'UltimoReajuste')
        temp = temp.join(cen,on = 'ReajusteDataBase',rsuffix = '_base')
        temp[f'{i}_base'] = temp[f'{i}_base'].fillna(temp['ipca_base'])
        temp[f'{i}'] = temp[f'{i}'].fillna(temp['ipca'])
        temp['ValorReajustadoCalculado'] = (temp[str(i)] / temp[f'{i}_base']) * temp['Valor']
        temp['TotalCalculado'] = temp['QuantSolicitada'] * temp['ValorReajustadoCalculado']
        temp = temp[['TotalCalculado','Competencia','Total']].groupby('Competencia').sum()
        temp['Diferenca'] = temp['TotalCalculado'] - temp['Total']
        cenarios.append(temp['Diferenca'].rename(i))
        if len(cenarios) % 100 == 0:
            print(len(cenarios))
    df = pd.concat(cenarios,axis = 1).cumsum()
    return df.set_index(df.index.to_timestamp())

def risco_mso(ipca,mso):
    size = 10000
    ipca['date'] = ipca['date'].apply(lambda x: datetime.datetime(int(x.split('-')[0]),int(x.split('-')[1]),1))
    ipca = ipca.set_index('date')
    std = ipca['std'].iloc[0]
    cen_df = (pd.concat([pd.Series(np.random.normal(scale = std,size = len(ipca)),index = ipca.index) for _ in range(size)],axis = 1).cumsum().T + ipca['prediction']).T
    anos = pd.Series(mso.index).apply(lambda x: x.year).unique()
    cenarios = []
    true_ipca = sgs.get({'IPCA_change':433},start = '2000-01-01',end = ipca.index[0] - datetime.timedelta(days = 1))
    true_ipca['ipca'] = [valor for valor in absolute(true_ipca['IPCA_change'],1598.41)]
    cen_df = pd.concat([pd.DataFrame({i:true_ipca['ipca'].values for i in range(size)},index = true_ipca.index),cen_df])
    for col in cen_df.columns:
        cen = cen_df[col]
        inflacoes = []
        for ano in anos:
            inicio = cen[datetime.datetime(ano,1,1)]
            fim = cen[datetime.datetime(ano + 1,1,1)]
            inflacoes += [(fim - inicio) / inicio] * 12
        inflacoes = np.array(inflacoes) + 1
        cenarios.append((mso['real'] * inflacoes) - mso['calc'])
        if len(cenarios) % 100 == 0:
            print(len(cenarios))
    df = pd.concat(cenarios,axis = 1).loc[datetime.datetime.today():].cumsum()
    return df
        

def return_cenarios_risco(risco):
    size = 1000
    risco_pred = risco
    if risco in ['TRADING','MSO']:
        risco_pred = 'INFLACAO'
    df_risco = retrieve_forecast(risco_pred)
    if risco == 'JUROS':
        selic = df_risco
        std = 0.024788480554999645
        selic['Período'] = selic['date'].apply(lambda x: pd.Period(x))
        selic = selic.set_index('Período')
        cen_df = (pd.concat([pd.Series(np.random.normal(scale = std,size = len(selic)),index = selic.index) for _ in range(size)],axis = 1).cumsum().T + selic['prediction']).T
    if risco == 'CAMBIO':
        cambio = df_risco
        std = 0.1281024428289958
        cambio['Período'] = cambio['date'].apply(lambda x: pd.Period(x))
        cambio = cambio.set_index('Período')
        cen_df = (pd.concat([pd.Series(np.random.normal(scale = std,size = len(cambio)),index = cambio.index) for _ in range(size)],axis = 1).cumsum().T + cambio['prediction']).T
    if risco in ['INFLACAO','TRADING','MSO']:
        ipca = df_risco
        ipca['date'] = ipca['date'].apply(lambda x: pd.Period(x))
        ipca = ipca.set_index('date')
        std = 16.80103821134177
        cen_df = (pd.concat([pd.Series(np.random.normal(0,std,len(ipca)),index = ipca.index) for i in range(size)],axis = 1).cumsum().T + ipca['prediction']).T
    if risco == 'GSF':
        gsf = df_risco
        std = 0.136
        gsf['Período'] = gsf['date'].apply(lambda x: pd.Period(x))
        gsf = gsf.set_index('Período')
        cen_df = pd.concat([(gsf['prediction'].rename(i) + pd.Series(np.random.normal(scale = std,size = len(gsf)),index = gsf.index)) for i in range(size)],axis = 1)
    return cen_df.set_index(cen_df.index.to_timestamp())

def calculate_cenarios(risco,df_risco = pd.DataFrame()):
    try:
        risco_pred = risco
        if risco in ['TRADING','MSO']:
            risco_pred = 'INFLACAO'
        if df_risco.empty:
            df_risco = retrieve_forecast(risco_pred)
        if risco == 'JUROS':
            dcf = read_cash_dcf()
            cen_df = risco_juros(df_risco,dcf)
        if risco == 'CAMBIO':
            rp = read_rp()
            cen_df = risco_cambio(df_risco,rp)
        if risco == 'INFLACAO':
            dfs = read_deb_ipca()
            cen_df = risco_ipca(df_risco,dfs)
        if risco == 'GSF':
            cen_df = risco_generico(df_risco)
        if risco == 'TRADING':
            con = read_contracts()
            cen_df = risco_trading(df_risco,con)
        if risco == 'MSO':
            mso = read_mso()
            cen_df = risco_mso(df_risco,mso)
        risc_df = pd.DataFrame(index = cen_df.index)
        risc_df.index.name = 'date'
        cen_df_resumed = pd.concat([cen_df.apply(lambda x: np.percentile(x,i),axis = 1) for i in range(1,100)],axis = 1).rename({i:'{}%'.format(i+1) for i in range(99)},axis = 1)
        risc_df['worst'] = cen_df_resumed['5%']
        risc_df['base'] = cen_df_resumed['50%']
        risc_df['best'] = cen_df_resumed['95%']
        risc_df['prob_worst'] = 0.05
        risc_df['prob_base'] = 0.9
        risc_df['prob_best'] = 0.05
        risc_df['probabilidade_impacto_negativo'] = cen_df.apply(lambda x: x < 0).sum(axis = 1) / 10000
        risc_df.to_csv(os.path.join(ctg_temp_path,f'risco_{risco}.csv'))
        risc_df.to_csv(os.path.join(downloads_path,f'risco_{risco}.csv'),sep = ';',decimal = ',')
        db.fetc_risco(risco,risc_df)
        pool = Pool(1)
        pool.apply_async(requests.get, args=[url_vini],callback=on_success, error_callback=on_error)
        upload_file(risco,'risco')
        cen_df_resumed.to_csv(os.path.join(ctg_temp_path,f'cenarios_{risco}.csv'))
        cen_df_resumed.to_csv(os.path.join(downloads_path,f'cenarios_{risco}.csv'),sep = ';',decimal = ',')
        upload_file(risco,'cenarios')
        return cen_df.drop_duplicates()
    except Exception as e:
        error(e)
        print(e)

def calculate_all():
    for risco in ['JUROS','CAMBIO','INFLACAO','TRADING','MSO']:
        calculate_cenarios(risco)