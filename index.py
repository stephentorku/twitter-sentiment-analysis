from select import select
from flask import Flask, url_for, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        company_name = request.form['company_name']
        date = request.form['sentiment_date']
        values = get_sentiments(date, company_name)
        return render_template("index.html", vals=values)
    values = get_sentiments("2022-04-09", "amazon")
    return render_template("index.html", vals=values)


if __name__ == "__main__":
    app.run()


#not found gives an error