from numpy import array
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from datetime import datetime


def stockchart(symbol,date):
    arr_num=[]
    ts = TimeSeries(key = 'P6WMOOB9UD8YMDTB', output_format = 'csv')

    #download the csv
    totalData = ts.get_intraday(symbol = symbol, interval = '30min', outputsize='full')

    #csv --> dataframe
    df = pd.DataFrame(list(totalData[0]))

    #setup of column and index
    header_row=0
    df.columns = df.iloc[header_row]
    df = df.drop(header_row)
    # df.set_index('time', inplace=False)
    df['Time'] = pd.to_datetime(df['timestamp']).dt.time
    df['Date'] =pd.to_datetime(df['timestamp']).dt.date
    dateT=datetime.strptime(date, '%Y-%m-%d').date()

    for i in df.index:
        if(df['Date'][i] == dateT):
            arr_num.append({"label":str(df['Time'][i]), "value":df['close'][i]})
    return arr_num
symbol=input("Enter symbol name:") 
date=input("Enter the start date:")
stockchart(symbol,date)   