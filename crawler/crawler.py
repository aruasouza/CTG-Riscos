from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from time import sleep
from retry import retry
import glob
from pathlib import Path
import os
from azure.datalake.store import core, lib, multithread
import pandas as pd
from datetime import datetime
from key import *

link_preco = 'https://tableaupub.ccee.org.br/t/CCEE/views/PreoHistrico/HistricodoPreoHorrio?%3Aembed=y&%3AshowVizHome=no&%3Ahost_url=https%3A%2F%2Ftableaupub.ccee.org.br%2F&%3Aembed_code_version=3&%3Atabs=no&%3Atoolbar=yes&%3Aiid=4&%3AisGuestRedirectFromVizportal=y&%3Adisplay_count=no&%3Aorigin=viz_share_link&%3AshowShareOptions=false&%3Aalerts=no&%3Arefresh=yes&%3Adisplay_spinner=no&%3AloadOrderID=1'
link_gsf = 'https://tableaupub.ccee.org.br/t/CCEE/views/PainelGerao/MRE?%3Aembed=y&%3AshowVizHome=no&%3Ahost_url=https%3A%2F%2Ftableaupub.ccee.org.br%2F&%3Aembed_code_version=3&%3Atabs=yes&%3Atoolbar=yes&%3AshowAppBanner=false&%3Adisplay_spinner=no&iframeSizedToWindow=true&%3AloadOrderID=0'
driver = webdriver.Chrome()
downloads_path = str(Path.home() / "Downloads")

@retry(tries = 5,delay = 1)
def find_and_click(selector,text,index = 0):
    driver.find_elements(selector,text)[index].click()

@retry(tries = 5,delay = 1)
def find_and_print(selector,text):
    elements = driver.find_elements(selector,text)
    if elements == []:
        raise ValueError
    for element in elements:
        print(element.get_attribute('class'))

@retry(tries = 5,delay = 1)
def upload_file_to_directory(file_name,directory):
    multithread.ADLUploader(adlsFileSystemClient, lpath=file_name,
        rpath=f'{directory}/{file_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    os.remove(file_name)

def get_last_file(name):
    path = os.path.join(downloads_path,name + '*')
    files = glob.glob(path)
    files.sort(key = lambda x: len(x))
    return files[-1]

driver.get(link_preco)

find_and_click(By.CSS_SELECTOR,'div.tabStoryPointContent.tab-widget',2)
find_and_click(By.CSS_SELECTOR,'div#download-ToolbarButton.tabToolbarButton.tab-widget.download')
find_and_click(By.CSS_SELECTOR,"button[class = 'f1odzkbq low-density']",2)
find_and_click(By.CSS_SELECTOR,"div[class = 'hidden-icon-wrapper_f72siau']",1)
sleep(1)
find_and_click(By.CSS_SELECTOR,"button[class = 'fycmrtt low-density']")
sleep(1)

driver.get(link_gsf)

find_and_click(By.CSS_SELECTOR,"div[class = 'tabToolbarButton tab-widget download']")
find_and_click(By.CSS_SELECTOR,"button[class = 'f1odzkbq low-density']",2)
find_and_click(By.CSS_SELECTOR,"div[class = 'hidden-icon-wrapper_f72siau']",7)
sleep(1)
find_and_click(By.CSS_SELECTOR,"button[class = 'fycmrtt low-density']")
sleep(3)

RESOURCE = 'https://datalake.azure.net/'
adlsAccountName = 'deepenctg'

adlCreds = lib.auth(tenant_id = tenant,
                client_secret = client_secret,
                client_id = client_id,
                resource = RESOURCE)

adlsFileSystemClient = core.AzureDLFileSystem(adlCreds, store_name=adlsAccountName)

file_precos = get_last_file('Média Mensal Comercial por Submercado')

mapa = {'janeiro':1,'fevereiro':2,'março':3,'abril':4,'maio':5,'junho':6,'julho':7,'agosto':8,'setembro':9,'outubro':10,'novembro':11,'dezembro':12}
df = pd.read_excel(file_precos,header = 1,index_col = 'Submercado').T.reset_index()
df['index'] = df['index'].apply(lambda x: datetime(int(x.split()[2]),mapa[x.split()[0]],1))
df = df.set_index('index')
df.columns.name = None
df.to_csv('preco_energia.csv')
upload_file_to_directory('preco_energia.csv','DataLakeRiscoECompliance/DadosEnergiaCCEE')

file_gsf = get_last_file('MRE - Geração x Garantia Física - Mês')

mapa_meses = {'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12}
new_names = {'Garantia física no centro de gravidade MW médios (GFIS_2p,j)':'Garantia Física',
    'Geração no Centro de Gravidade - MW médios (Gp,j)':'Geração'}
df = pd.read_excel(file_gsf).set_index('Unnamed: 0').T.rename(new_names,axis = 1)
df.columns.name = None
df = df.reset_index()
df['index'] = pd.to_datetime(df['index'].apply(lambda x: str(mapa_meses[x[:3]]) + x[3] + '20' + x[4:]),format = '%m/%Y')
df = df.set_index('index')
df.to_csv('gsf.csv')
upload_file_to_directory('gsf.csv','DataLakeRiscoECompliance/DadosEnergiaCCEE')