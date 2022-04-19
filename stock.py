from alpha_vantage.timeseries import TimeSeries 
import matplotlib.pyplot as plt 
import sys

def stockchart(symbol,date):
    ts = TimeSeries(key='P6WMOOB9UD8YMDTB', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval='30min', outputsize='full')
    data_date_changed = data.loc[date]
    # data_date_changed['4. close'].plot()
    # data_date_changed.loc[date].plot()
    print(data_date_changed['4. close'])
    # plt.title('Stock chart')
    # plt.show()

symbol=input("Enter symbol name:") 
date=input("Enter the start date:")
stockchart(symbol,date)