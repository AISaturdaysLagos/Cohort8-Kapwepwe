# -*- coding: utf-8 -*-
"""Restaurant-model-building.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AAWeryusFRhP59u7UHD1pa1fOYhDouaX
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

nltk.download('stopwords')
nltk.download('punkt')

restaurants_init = pd.read_csv('clean_lagos_restaurants.csv')

# removing manager feedback
feedback_string = 'Thank you for'
manager = restaurants_init.loc[restaurants_init['review_text'].str.contains(feedback_string)]

restaurants_init = restaurants_init.drop(manager.index.values, axis=0)

restaurants_init.drop(columns='Unnamed: 0', inplace=True)

# converting all text to lowercase
restaurants_init['review_text'] = restaurants_init['review_text'].str.lower()

restaurants_init['review_title'] = restaurants_init['review_title'].str.lower()

# converting emojis to words
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
# Converting emojis to words
def convert_emojis(text):
  for emot in UNICODE_EMOJI:
    text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
  return text


# Passing both functions to the review text and review title
restaurants_init['review_text'] = restaurants_init['review_text'].apply(convert_emojis)
restaurants_init['review_title'] = restaurants_init['review_title'].apply(convert_emojis)


# translating to english only
from langdetect import detect
from deep_translator import GoogleTranslator

for rec in restaurants_init['review_text']:
  result_lang = detect(rec)
  if result_lang == 'en':
    rec = rec
  else:
    rec = GoogleTranslator(target='en').translate(str(rec))
    # rec = translator.translate(rec, lang_src=result_lang, lang_tgt='en')

# removing stopwords in the review text
stop_words = stopwords.words('english')
restaurants_init['review_text'] = restaurants_init['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

# removing stopwords in review title
restaurants_init['review_title'] = restaurants_init['review_title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

# removing punctuation
punctuation_and_symbols = "`~!@#$%^&*()-_+=[\]{|};:'<>.,?/\n"
for _ in range(len(punctuation_and_symbols)):
  restaurants_init['review_text'] = restaurants_init['review_text'].str.replace(punctuation_and_symbols[_], ' ', regex=True)
  restaurants_init['review_title'] = restaurants_init['review_title'].str.replace(punctuation_and_symbols[_], ' ', regex=True)
  restaurants_init['review_text'] = restaurants_init['review_text'].str.replace("  ", " ")
  restaurants_init['review_title'] = restaurants_init['review_title'].str.replace("  ", " ")

# lemmatizing
from textblob import Word

restaurants_init['review_text'] = restaurants_init['review_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
restaurants_init['review_title'] = restaurants_init['review_title'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

emptyline=[]
for row in restaurants_init['review_text']:
 vs = analyzer.polarity_scores(row)
 emptyline.append(vs)

df_sentiments = pd.DataFrame(emptyline)

restaurants_final = pd.concat([restaurants_init.reset_index(drop=True), df_sentiments], axis=1)

restaurants_final['Sentiment'] = np.where(restaurants_final['compound'] > 0.05, 'Positive',
         (np.where(restaurants_final['compound'] < -0.05, 'Negative', 'Neutral')))

restaurants_final = restaurants_final.drop(['neg', 'neu', 'pos', 'compound'], axis=1)

result = restaurants_final.groupby('restaurant_name')['Sentiment'].value_counts().unstack()

result.fillna(0, inplace=True)

result['opinion'] = result[['Negative','Neutral', 'Positive']].idxmax(axis=1)

result = result.reset_index()


import streamlit as st

st.title("Lagos Restaurants Sentiment Analyser App")
st.write("Get an accurate feel of what people think about a restaurant's service!")

form = st.form(key='sentiment-form')
user_input = form.text_area("Enter a restaurant's name")
submit = form.form_submit_button('Submit')

condition = user_input in result['restaurant_name'].values

if submit:
  if condition == True:
    label = user_input
    whole_row = result[result['restaurant_name'] == label]
    score = whole_row['opinion'].values[0]
    if score == 'Positive':
      st.success(f'Many customer find {label} a good place to spend their money!')
    elif score == 'Negative' or score == 'Neutral':
      st.success(f'The average customer finds {label} not so great for eating out. Maybe try somewhere else?')
  else:
      st.error(f'{user_input} is not in our database, we apologize about that.')