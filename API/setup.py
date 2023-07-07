from azure.datalake.store import core, lib
import time
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