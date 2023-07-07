url = 'http://191.238.218.31:5000'

import requests
import pandas as pd
from setup import *
from retry import retry
from key import admin_pass

# Função que devolve o error e concatena no arquivo de log
@retry(tries = 5,delay = 1)
def error(e):
    log = pd.read_csv(logfile_path)
    log = pd.concat([log,pd.DataFrame({'time':[datetime.now()],'output':['erro'],'error':[repr(e)]})])
    log.to_csv(logfile_path,index = False)
    try:
        multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_path,
            rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    except Exception as err:
        print(err)
        print('Tentando novamente em 1 segundo...')
    finally:
        raise e

@retry(tries = 5,delay = 1)
def success(name,output):
    time = datetime.now()
    time_str = str(time).replace('.','-').replace(':','-').replace(' ','-')
    file_name = f'{name}_{time_str}.csv'
    file_path = os.path.join(ctg_temp_path,file_name)
    output.index.name = 'date'
    output.to_csv(file_path)
    output.to_csv(os.path.join(downloads_path,file_name))
    upload_file_to_directory(file_path,f'DataLakeRiscoECompliance/PrevisionData/Variables/{name}/AI',file_name)
    log = pd.read_csv(logfile_path)
    log = pd.concat([log,pd.DataFrame({'time':[time],'output':[file_name],'error':['no errors']})])
    log.to_csv(logfile_path,index = False)
    records = pd.read_csv(records_path)
    records = pd.concat([records,pd.DataFrame({'file_name':[file_name],'origin':'AI'})])
    records.to_csv(records_path,index = False)
    try:
        multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_path,
            rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
        multithread.ADLUploader(adlsFileSystemClient, lpath=records_path,
            rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    except Exception as e:
        print(e)
        print('Tentando novamente em 1 segundo...')
        raise e

@retry(tries = 5,delay = 1)
def upload_file_to_directory(file_path,directory,file_name):
    try:
        multithread.ADLUploader(adlsFileSystemClient, lpath=file_path,
            rpath=f'{directory}/{file_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    except Exception as e:
        print(e)
        print('Tentando novamente em 1 segundo...')
        raise e

def predict_ipca():
    global run_status
    global data_for_plotting
    req = requests.get(url + '/ipca',auth = ('deepen',admin_pass))
    if req.status_code == 200:
        data = req.json()
        pred = data['pred']
        past = data['past']
        pred_df = pd.DataFrame(pred['data'],pred['index'],pred['columns'])
        past_df = pd.DataFrame(past['data'],past['index'],past['columns'])
        success('INFLACAO',pred_df)
        data_for_plotting = pd.concat([past_df,pred_df])
        data_for_plotting = data_for_plotting.set_index(pd.to_datetime(data_for_plotting.index,format = '%Y-%m')).drop('std',axis = 1)
        run_status = 'O forecast foi gerado e enviado com sucesso para a nuvem'
    else:
        message = req.json()['error']
        e = ValueError(message)
        error(e)
        run_status = e

def predict_cambio():
    global run_status
    global data_for_plotting
    req = requests.get(url + '/cambio',auth = ('deepen',admin_pass))
    if req.status_code == 200:
        data = req.json()
        pred = data['pred']
        past = data['past']
        pred_df = pd.DataFrame(pred['data'],pred['index'],pred['columns'])
        past_df = pd.DataFrame(past['data'],past['index'],past['columns'])
        success('CAMBIO',pred_df)
        data_for_plotting = pd.concat([past_df,pred_df])
        data_for_plotting = data_for_plotting.set_index(pd.to_datetime(data_for_plotting.index,format = '%Y-%m')).drop('std',axis = 1)
        run_status = 'O forecast foi gerado e enviado com sucesso para a nuvem'
    else:
        message = req.json()['error']
        e = ValueError(message)
        error(e)
        run_status = e

def predict_cdi():
    global run_status
    global data_for_plotting
    req = requests.get(url + '/cdi',auth = ('deepen',admin_pass))
    if req.status_code == 200:
        data = req.json()
        pred = data['pred']
        past = data['past']
        pred_df = pd.DataFrame(pred['data'],pred['index'],pred['columns'])
        past_df = pd.DataFrame(past['data'],past['index'],past['columns'])
        success('JUROS',pred_df)
        data_for_plotting = pd.concat([past_df,pred_df])
        data_for_plotting = data_for_plotting.set_index(pd.to_datetime(data_for_plotting.index,format = '%Y-%m')).drop('std',axis = 1)
        run_status = 'O forecast foi gerado e enviado com sucesso para a nuvem'
    else:
        message = req.json()['error']
        e = ValueError(message)
        error(e)
        run_status = e

def predict_gsf():
    global run_status
    global data_for_plotting
    req = requests.get(url + '/gsf',auth = ('deepen',admin_pass))
    if req.status_code == 200:
        data = req.json()
        pred = data['pred']
        past = data['past']
        pred_df = pd.DataFrame(pred['data'],pred['index'],pred['columns'])
        past_df = pd.DataFrame(past['data'],past['index'],past['columns'])
        success('GSF',pred_df)
        data_for_plotting = pd.concat([past_df,pred_df])
        data_for_plotting = data_for_plotting.set_index(pd.to_datetime(data_for_plotting.index,format = '%Y-%m')).drop('std',axis = 1)
        run_status = 'O forecast foi gerado e enviado com sucesso para a nuvem'
    else:
        message = req.json()['error']
        e = ValueError(message)
        error(e)
        run_status = e