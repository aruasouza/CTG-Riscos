def main():
    print('----------')
    print('*** CTG Riscos ***')
    print('----------')
    print('WARNING:','Fechar esta tela fechará a aplicação.')
    print('----------')
    print('Carregando, aguarde...')
    print('\n')
    print('Importando bibliotecas...')
    from tkinter import ttk
    import tkinter as tk
    from ttkthemes import ThemedTk
    from tkinter import font,Menu
    from tkinter import filedialog
    print('tkinter')
    import threading
    print('threading')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    print('matplotlib')
    from azure.datalake.store import multithread
    print('azure')
    import pandas as pd
    print('pandas')
    from datetime import datetime
    print('datetime')
    import re
    print('re')
    import numpy as np
    print('numpy')
    import os
    print('os')
    import tempfile
    print('tempfile')
    import shutil
    print('shutil')
    print('\n')
    temp = tempfile.gettempdir()
    ctg_temp_path = os.path.join(temp,'ctgriscos')
    if os.path.exists(ctg_temp_path):
        shutil.rmtree(ctg_temp_path)
        os.mkdir(ctg_temp_path)
    else:
        os.mkdir(ctg_temp_path)
    import models
    from setup import upload_file_to_directory,logfile_name,adlsFileSystemClient,logfile_path,records_path,downloads_path
    import montecarlo
    import budget
    import cenarios_personalizados as cp

    def upload_file(Risco):
        create_upload_window()
        file_path = filedialog.askopenfilename()
        try:
            output = pd.read_csv(file_path,sep = ';',decimal = ',')
        except Exception:
            ttk.Label(root,text = 'Erro: O arquivo selecionado não é do formato correto.').place(relx=0.5, rely=0.2, anchor='center')
            return
        if len(output) > 72:
            ttk.Label(root,text = 'Erro: O tamanho do arquivo excede o limite máximo').place(relx=0.5, rely=0.2, anchor='center')
            return
        if list(output.columns) != ['date', 'prediction', 'std']:
            ttk.Label(root,text = 'Erro: As colunas do arquivo devem ser (nessa ordem): date, prediction, superior, inferior, std').place(relx=0.5, rely=0.2, anchor='center')
            return
        date_sample = output.loc[0,'date'] 
        if not re.match("^\d{4}-\d{2}$", date_sample):
            ttk.Label(root,text = 'Erro: As datas não estão no formato correto (YYYY-mm). Exemplo: 2020-04').place(relx=0.5, rely=0.2, anchor='center')
            return
        time = datetime.now()
        time_str = str(time).replace('.','-').replace(':','-').replace(' ','-')
        file_name = f'{Risco}_{time_str}.csv'
        # Colocando o output para csv e encaminhando-o para o data lake
        save_path = os.path.join(ctg_temp_path,'output.csv')
        output.to_csv(save_path)
        upload_file_to_directory(save_path,f'DataLakeRiscoECompliance/PrevisionData/Variables/{Risco}/Manual',file_name)
        log = pd.read_csv(logfile_path)
        log = pd.concat([log,pd.DataFrame({'time':[time],'output':[file_name],'error':['no errors']})])
        log.to_csv(logfile_path,index = False)
        try:
            multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_path,
                rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)    
            records = pd.read_csv(records_path)
            records = pd.concat([records,pd.DataFrame({'file_name':[file_name],'origin':['Manual']})])
            records.to_csv(records_path,index = False)
            multithread.ADLUploader(adlsFileSystemClient, lpath=records_path,
                rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
            ttk.Label(root,text = 'O arquivo foi enviado com sucesso para a nuvem.').place(relx=0.5, rely=0.2, anchor='center')
        except Exception as e:
            ttk.Label(root,text = str(e)).place(relx=0.5, rely=0.2, anchor='center')

    def upload_cen_file(Risco):
        create_upload_cen_window()
        file_path = filedialog.askopenfilename()
        try:
            output = pd.read_csv(file_path,sep = ';',decimal = ',')
        except Exception:
            ttk.Label(root,text = 'Erro: O arquivo selecionado não é do formato correto.').place(relx=0.5, rely=0.2, anchor='center')
            return
        if len(output) > 72:
            ttk.Label(root,text = 'Erro: O tamanho do arquivo excede o limite máximo').place(relx=0.5, rely=0.2, anchor='center')
            return
        if list(output.columns) != ['date', 'worst', 'base', 'best', 'prob_worst','prob_base','prob_best']:
            ttk.Label(root,text = 'Erro: As colunas do arquivo devem ser (nessa ordem): date, worst, base, best, prob_worst, prob_base, prob_best').place(relx=0.5, rely=0.2, anchor='center')
            return
        date_sample = output.loc[0,'date']
        if not re.match("^\d{4}-\d{2}$", date_sample):
            ttk.Label(root,text = 'Erro: As datas não estão no formato correto (YYYY-mm). Exemplo: 2020-04').place(relx=0.5, rely=0.2, anchor='center')
            return
        terminate_window()
        ttk.Label(root,text = 'Calculando. Isso pode levar alguns minutos.').place(relx=0.5, rely=0.5, anchor='center')
        output = output.set_index('date')
        thread = threading.Thread(target = lambda: thread_upload_cen(Risco,output))
        thread.start()

    def download_risc_files():
        try:
            for risco in ['JUROS','CAMBIO','INFLACAO','TRADING','MSO','GSF']:
                time = str(datetime.now()).split('.')[0].replace(':','-')
                multithread.ADLDownloader(adlsFileSystemClient, lpath=os.path.join(downloads_path,f'{risco} {time}.csv'),
                rpath=f'DataLakeRiscoECompliance/RISCOS/{risco}/risco_{risco}.csv', nthreads=64, 
                overwrite=True, buffersize=4194304, blocksize=4194304)
            terminate_window()
            ttk.Label(root,text = 'Sucesso. Os arquivos foram enviados para a pasta "Downloads"').place(relx=0.5, rely=0.5, anchor='center')
            ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
        except Exception as e:
            create_error_window(e)

    def thread_upload_cen(Risco,output):
        try:
            cen = cp.calculate(Risco,output)
            terminate_window()
            ttk.Label(root,text = 'Sucesso. Os cenários foram enviados para a pasta "Downloads"').place(relx=0.5, rely=0.4, anchor='center')
            ttk.Button(root,text = 'Visualizar',command = lambda: show_simulation(cen,1,Risco,False)).place(relx=0.5,rely=0.6,anchor='center')
        except Exception as e:
            create_error_window(e)
            
    def show_forecast():
        fig = Figure(figsize = (10,7),dpi = 100)
        fig.add_subplot(111).plot(models.data_for_plotting)
        canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

    def show_simulation(cen_df,alpha = 0.1,risco = None,labels = True):
        fig = Figure(figsize = (5,3),dpi = 100)
        if risco == None:
            risco = montecarlo.main_info['risco']
        if risco == 'IPCA':
            name = 'Contratos Financeiros (Inflação)'
        elif risco == 'JUROS':
            name = 'Contratos Finaceiros (Juros)'
        elif risco == 'TRADING':
            name = 'Contratos de Energia (Inflação)'
        elif risco == 'CAMBIO':
            name = 'Contratos Finaceiros (Câmbio)'
        elif risco == 'MSO':
            name = 'MSO (Inflação)'
        else:
            name = risco
        ax = fig.add_subplot(111)
        for cenario in cen_df.columns[:1000]:
            ax.plot(cen_df.index,cen_df[cenario],alpha = alpha,color = 'red')
        ax.set_title(f'Cenários {name}')
        canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
        row = cen_df.iloc[-1]
        pior,medio = min(0,int(np.percentile(row.values,5))),min(0,int(np.percentile(row.values,50)))
        if labels:
            ttk.Label(root,text = 'Pior cenário (5%): Prejuízo de R$ {valor:_},00 ou mais.'.format(valor = abs(pior)).replace('_','.')).place(relx=0.5, rely=0.80, anchor='center')
            ttk.Label(root,text = 'Cenário Médio: Prejuízo de R$ {valor:_},00.'.format(valor = abs(medio)).replace('_','.')).place(relx=0.5, rely=0.84, anchor='center')

    def create_done_window():
        terminate_window()
        ttk.Label(root,text = models.run_status).place(relx=0.5, rely=0.4, anchor='center')
        if models.run_status == 'O forecast foi gerado e enviado com sucesso para a nuvem':
            ttk.Label(root,text = 'Também está disponível na pasta "Downloads"').place(relx=0.5, rely=0.5, anchor='center')
            ttk.Button(root,text = 'Visualizar',command = show_forecast).place(relx=0.5,rely=0.6,anchor='center')
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

    def thread_ipca():
        try:
            models.predict_ipca()
            create_done_window()
        except Exception as e:
            create_error_window(e)

    def thread_cambio():
        try:
            models.predict_cambio()
            create_done_window()
        except Exception as e:
            create_error_window(e)

    def thread_cdi():
        try:
            models.predict_cdi()
            create_done_window()
        except Exception as e:
            create_error_window(e)

    def thread_gsf():
        try:
            models.predict_gsf()
            create_done_window()
        except Exception as e:
            create_error_window(e)

    def forecast_ipca():
        terminate_window()
        ttk.Label(root,text = 'Calculando. Isso pode levar alguns minutos.').place(relx=0.5, rely=0.5, anchor='center')
        thread = threading.Thread(target = thread_ipca)
        thread.start()

    def forecast_cambio():
        terminate_window()
        ttk.Label(root,text = 'Calculando. Isso pode levar alguns minutos.').place(relx=0.5, rely=0.5, anchor='center')
        thread = threading.Thread(target = thread_cambio)
        thread.start()

    def forecast_cdi():
        terminate_window()
        ttk.Label(root,text = 'Calculando. Isso pode levar alguns minutos.').place(relx=0.5, rely=0.5, anchor='center')
        thread = threading.Thread(target = thread_cdi)
        thread.start()

    def forecast_gsf():
        terminate_window()
        ttk.Label(root,text = 'Calculando. Isso pode levar alguns minutos.').place(relx=0.5, rely=0.5, anchor='center')
        thread = threading.Thread(target = thread_gsf)
        thread.start()

    def calcular_cenarios(index):
        terminate_window()
        ttk.Label(root,text = 'Calculando. Isso pode levar alguns minutos.').place(relx=0.5, rely=0.5, anchor='center')
        thread = threading.Thread(target = lambda: thread_cenarios(index))
        thread.start()

    def thread_cenarios(index):
        try:
            montecarlo.find_datas(index)
            cen_df = budget.calculate_cenarios(montecarlo.main_info['risco'],montecarlo.main_dataframe)
            simulation_done_window(cen_df)
        except Exception as e:
            create_error_window(e)

    def simulation_done_window(cen_df):
        terminate_window()
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
        ttk.Button(root,text = 'Visualizar',command = lambda: show_simulation(cen_df)).place(relx=0.5,rely=0.5,anchor='center')

    def get_file_names(risco):
        terminate_window()
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
        try:
            montecarlo.find_files(risco)
            info = montecarlo.main_info
            menu = Menu(root)
            for i in range(info['size'] - 1,-1,-1):
                menu.add_command(label = info['full_strings'][i],command = lambda i=i: calcular_cenarios(i))
            ttk.Menubutton(root,text = 'Selecionar Arquivo',menu = menu).place(relx=0.5, rely=0.5, anchor='center')
        except Exception as e:
            create_error_window(e)

    def create_forecast_window():
        terminate_window()
        ttk.Button(root, text="IPCA", command=forecast_ipca).place(relx=0.5, rely=0.3, anchor='center')
        ttk.Button(root, text="Câmbio", command=forecast_cambio).place(relx=0.5, rely=0.4, anchor='center')
        ttk.Button(root, text="CDI", command=forecast_cdi).place(relx=0.5, rely=0.5, anchor='center')
        ttk.Button(root, text="GSF", command=forecast_gsf).place(relx=0.5, rely=0.6, anchor='center')
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

    def create_upload_window():
        terminate_window()
        ttk.Button(root, text="IPCA", command=lambda: upload_file('INFLACAO')).place(relx=0.5, rely=0.3, anchor='center')
        ttk.Button(root, text="Câmbio", command=lambda: upload_file('CAMBIO')).place(relx=0.5, rely=0.4, anchor='center')
        ttk.Button(root, text="CDI", command=lambda: upload_file('JUROS')).place(relx=0.5, rely=0.5, anchor='center')
        ttk.Button(root, text="GSF", command=lambda: upload_file('GSF')).place(relx=0.5, rely=0.6, anchor='center')
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

    def create_upload_cen_window():
        terminate_window()
        ttk.Button(root, text='Contratos Financeiros (Inflação)', command=lambda: upload_cen_file('INFLACAO')).place(relx=0.5, rely=0.3, anchor='center')
        ttk.Button(root, text='Contratos Finaceiros (Câmbio)', command=lambda: upload_cen_file('CAMBIO')).place(relx=0.5, rely=0.4, anchor='center')
        ttk.Button(root, text='Contratos Finaceiros (Juros)', command=lambda: upload_cen_file('JUROS')).place(relx=0.5, rely=0.5, anchor='center')
        ttk.Button(root, text='Contratos de Energia (Inflação)', command=lambda: upload_cen_file('TRADING')).place(relx=0.5, rely=0.6, anchor='center')
        ttk.Button(root, text="MSO (Inflação)", command=lambda: upload_cen_file('MSO')).place(relx=0.5, rely=0.7, anchor='center')
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

    def create_simulador_window():
        terminate_window()
        ttk.Button(root, text='Contratos Financeiros (Inflação)', command=lambda: get_file_names('INFLACAO')).place(relx=0.5, rely=0.3, anchor='center')
        ttk.Button(root, text='Contratos Finaceiros (Câmbio)', command=lambda: get_file_names('CAMBIO')).place(relx=0.5, rely=0.4, anchor='center')
        ttk.Button(root, text='Contratos Finaceiros (Juros)', command=lambda: get_file_names('JUROS')).place(relx=0.5, rely=0.5, anchor='center')
        ttk.Button(root, text="GSF", command=lambda: get_file_names('GSF')).place(relx=0.5, rely=0.8, anchor='center')
        ttk.Button(root, text='Contratos de Energia (Inflação)', command=lambda: get_file_names('TRADING')).place(relx=0.5, rely=0.6, anchor='center')
        ttk.Button(root, text="MSO (Inflação)", command=lambda: get_file_names('MSO')).place(relx=0.5, rely=0.7, anchor='center')
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

    def create_main_window():
        terminate_window()
        ttk.Button(root, text="Gerar Forecasts", command=create_forecast_window).place(relx=0.5, rely=0.3, anchor='center')
        ttk.Button(root, text="Subir Forecast Personalizado", command=create_upload_window).place(relx=0.5, rely=0.4, anchor='center')
        ttk.Button(root, text="Simular Cenários", command=create_simulador_window).place(relx=0.5, rely=0.5, anchor='center')
        ttk.Button(root, text="Calcular Cenários Personalizados", command=create_upload_cen_window).place(relx=0.5, rely=0.6, anchor='center')
        ttk.Button(root, text="Baixar arquivos de Risco", command=download_risc_files).place(relx=0.5, rely=0.7, anchor='center')

    def create_error_window(e):
        terminate_window()
        ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
        ttk.Label(root,text = str(e)).place(relx=0.5, rely=0.5, anchor='center')
        raise e
    
    def terminate_window():
        for element in root.winfo_children():
            element.destroy()

    root = ThemedTk(theme="adapta")
    root.geometry("800x600")
    style = ttk.Style()
    custom_font = font.Font(family="Montserrat", size=15)
    style.configure("TButton", font = custom_font)

    create_main_window()

    root.mainloop()

main()