from azure.datalake.store import core, lib, multithread
import pandas as pd
from datetime import datetime
import os
import time
import tempfile
from pathlib import Path
from key import *

downloads_path = str(Path.home() / "Downloads")

temp = tempfile.gettempdir()
ctg_temp_path = os.path.join(temp,'ctgriscos')

for i in range(5):
    print('Estabelecendo conexão com o Azure')
    RESOURCE = 'https://datalake.azure.net/'
    adlsAccountName = 'deepenctg'
    today = datetime.now()
    logfile_name = f'log_{today.month}_{today.year}.csv'
    logfile_path = os.path.join(ctg_temp_path,logfile_name)
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

try:
    print('Verificando o LOG')
    multithread.ADLDownloader(adlsFileSystemClient, lpath=logfile_path, 
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)
    print('LOG obtido com sucesso')

except FileNotFoundError:
    pd.DataFrame({'time':[],'output':[],'error':[]}).to_csv(logfile_path,index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_path,
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    print('Novo LOG criado')

except Exception as e:
    print(e)
    print('\n')
    print('Erro na verificação do LOG, algumas funções não irão funcionar corretamente')

records_path = os.path.join(ctg_temp_path,'records.csv')
try:
    print('Verificando registros')
    multithread.ADLDownloader(adlsFileSystemClient, lpath=records_path,
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64,
        overwrite=True, buffersize=4194304, blocksize=4194304)
    print('Registros obtidos com sucesso')

except FileNotFoundError:
    pd.DataFrame({'file_name':[],'origin':[]}).to_csv(records_path,index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=records_path,
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    print('Novo arquivo de registros criado')
    print('WARNING: forecasts criados anteriormente não poderão ser utilizados')
    
except Exception as e:
    print(e)
    print('\n')
    print('Erro na verificação dos registros, algumas funções não irão funcionar corretamente')

def upload_file_to_directory(local_path,directory,file_name):
    multithread.ADLUploader(adlsFileSystemClient, lpath=local_path,
        rpath=f'{directory}/{file_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    

to_process = 'DataLakeRiscoECompliance/Sharepoint/files_to_process.txt'
file_links = []
try:
    print('Baixando lista de arquivos...')
    multithread.ADLDownloader(adlsFileSystemClient, lpath=os.path.join(ctg_temp_path,'files_to_process.txt'), 
                rpath=to_process, nthreads=64, 
                overwrite=True, buffersize=4194304, blocksize=4194304)
    print('Sucesso')
    with open(os.path.join(ctg_temp_path,'files_to_process.txt'),'r') as file:
        for line in file:
            file_links.append(line[:-1])
except Exception as e:
    print(e)
    print('\n')
    print('Erro no download do arquivo. Cenários de risco não poderão der calculados')


file_names = {}
for file in file_links:
    try:
        print('Baixando',file)
        multithread.ADLDownloader(adlsFileSystemClient, lpath=os.path.join(ctg_temp_path,file.split('/')[-1]), 
                rpath=file, nthreads=64, 
                overwrite=True, buffersize=4194304, blocksize=4194304)
        print('Sucesso')
        file_names[file.split('/')[-1].split('_')[0]] = file.split('/')[-1]
    except Exception as e:
        print(e)
        print('\n')
        print('Erro no download do arquivo. Cenários de risco não poderão der calculados')