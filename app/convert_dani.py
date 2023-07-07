import pandas as pd
import json
import time
from azure.datalake.store import core, lib, multithread
from key import *


for i in range(5):
    print('Estabelecendo conexão com o Azure')
    RESOURCE = 'https://datalake.azure.net/'
    adlsAccountName = 'deepenctg'
    try:
        adlCreds = lib.auth(tenant_id = tenant,
                    client_secret = client_secret,
                    client_id = client_id,
                    resource = RESOURCE)
        adlsFileSystemClient = core.AzureDLFileSystem(adlCreds, store_name=adlsAccountName)
        print('Sucesso')
        break
    except Exception as e:
        adlCreds = None
        adlsFileSystemClient = None
        print(e)
        print('Tentando novamente em 1 segundo')
        time.sleep(1)
        if i == 4:
            print('\n')
            print('ERRO: SEM CONEXÃO COM A AZURE')
            print('Fechando em 5 segundos')
            time.sleep(5)
            raise Exception

print('Obtendo contratos...')
adl = adlsFileSystemClient
files_list = list(filter(lambda x: (x.split('/')[-1][-8:-5] == 'wbc') and (x.split('/')[-1][-4:] == 'json'),adl.walk('LandingData/Comercial/wbc')))
files_list.sort()
name = files_list[-1]
with adl.open(name, 'rb') as f:
    js = json.load(f)
df = pd.DataFrame.from_records(js['Values'])
df.to_excel(name.split('/')[-1].split('.')[0] + '.xlsx')