from flask import Flask, url_for, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import json
import flask


app = Flask(__name__)

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)



def get_sentiments(input_date, company_name):
    #df = pd.read_csv("all-tweets.csv")
    df = pd.read_csv("all-tweets.csv")
    positive_sentiment_count=0
    negative_sentiment_count=0
    neutral_sentiment_count=0

    for i in range(len(df)):
        #check for case sensitivity
        #looping through dataframe
        if input_date in df.loc[i, "created_at"] and company_name in df.loc[i, "text"]:
            tweet_words=[]

            #preprocessing data
            for word in df.loc[i, "text"].split(' '):
                if word.startswith('@') and len(word) > 1:
                    word = '@user'
                elif word.startswith('https'):
                    word = "http"
                tweet_words.append(word)
            tweet_proc = " ".join(tweet_words)

            encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
            output = model(**encoded_tweet)
            scores = output[0][0].detach().numpy()

            #finding index of highest sentiment score
            max_score = np.max(scores)
            index_of_score = np.where(scores == max_score)
            index_of_score = index_of_score[0][0]

            #increase counter depending on type of sentiment
            if index_of_score ==0:
                negative_sentiment_count+=1
            elif index_of_score ==1:
                neutral_sentiment_count+=1
            else:
                positive_sentiment_count+=1

    #storing counts in array
    values = [negative_sentiment_count, positive_sentiment_count, neutral_sentiment_count]
    return values

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

@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template("index.html")

#endpoint for receiving and calculating information
@app.route('/get_data',  methods=['POST', 'GET'])
def get_data():
    if request.method == "POST":
        #lists for storing labels and stock prices from stock function
        labels=[]
        stock_list=[]

        #receiving data from form
        company_name = str(request.form['company_name'])
        sentiment_date = str(request.form['sentiment_date'])
        stock_date = str(request.form['stock_date'])

        #getting sentiment counts and stock values from functions using form data
        values = get_sentiments(sentiment_date, company_name)
        stock_values = stockchart(company_name, stock_date)


        for dic in stock_values:
            labels.append(dic['label'])
            stock_list.append(float(dic['value']))

        #sending data as JSON to endpoint
        return flask.jsonify({'payload':json.dumps({'data':list(reversed(stock_list)), 'labels':list(reversed(labels)), 'sentiments': values , "minimum":min(list(stock_list)), "maximum":max(list(stock_list)) })})

    else:
        labels = []
        stock_list=[]
        
        stock_values = stockchart("AMZN", "2022-04-18")
        for dic in stock_values:
            labels.append(dic['label'])
            stock_list.append(float(dic['value']))
        values = get_sentiments("2022-04-18", "AMZN")
        return flask.jsonify({'payload':json.dumps({'data':list(reversed(stock_list)), 'labels':list(reversed(labels)), 'sentiments': values, "minimum":min(list(stock_list)), "maximum":max(list(stock_list))} )})


if __name__ == "__main__":
    #app.run(host='0.0.0.0', debug=True, port=5000)
    app.run(debug=True)


#not found gives an error