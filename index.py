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
    df = pd.read_csv("tweets.csv")
    positive_sentiment_count=0
    negative_sentiment_count=0
    neutral_sentiment_count=0
    # input_date = "2022-04-09"
    # company_name = "TSLA"
    created_at = df.loc[df['created_at'].str.contains(input_date, case=False)]
    for i in range(len(df)):
        #check for case sensitivity
        if input_date in df.loc[i, "created_at"] and company_name in df.loc[i, "text"]:
            tweet_words=[]
            # words = df.loc[i, "text"].split(' ')
            # print("words")
            # print(words)
            for word in df.loc[i, "text"].split(' '):
                if word.startswith('@') and len(word) > 1:
                    word = '@user'
                elif word.startswith('https'):
                    word = "http"
                tweet_words.append(word)
            tweet_proc = " ".join(tweet_words)
            # print("tweet proc:")
            # print(tweet_proc)
            encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
            output = model(**encoded_tweet)
            scores = output[0][0].detach().numpy()
            # print("scores")
            # print(scores)
            max_score = np.max(scores)
            index_of_score = np.where(scores == max_score)
            index_of_score = index_of_score[0][0]
            # print("index")
            # print(index_of_score)
            if index_of_score ==0:
                negative_sentiment_count+=1
            elif index_of_score ==1:
                neutral_sentiment_count+=1
            else:
                positive_sentiment_count+=1
            
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

@app.route('/get_data',  methods=['POST', 'GET'])
def get_data():
    labels = []
    stock_list=[]
    if request.method == "POST":
        print("post")
        print(request.form['company_name'])
        print(request.form['sentiment_date'])
        print(request.form['stock_date'])
        company_name = request.form['company_name']
        sentiment_date = request.form['sentiment_date']
        stock_date = request.form['stock_date']
        values = get_sentiments(sentiment_date, company_name)
        stock_values = stockchart(company_name, stock_date)
        for dic in stock_values:
            labels.append(dic['label'])
            stock_list.append(float(dic['value']))
        return flask.jsonify({'payload':json.dumps({'data':list(reversed(stock_list)), 'labels':list(reversed(labels)), 'sentiments': values})})
    else:
        stock_values = stockchart("GOOG", "2022-04-12")
        for dic in stock_values:
            labels.append(dic['label'])
            stock_list.append(float(dic['value']))
        values = get_sentiments("2022-04-09", "TSLA")
        return flask.jsonify({'payload':json.dumps({'data':list(reversed(stock_list)), 'labels':list(reversed(labels)), 'sentiments': values})})


if __name__ == "__main__":
    app.run(debug=True)


#not found gives an error