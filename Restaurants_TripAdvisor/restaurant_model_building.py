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
import pandas as pd


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

dir_name = os.path.abspath(os.path.dirname(__file__))
location = os.path.join(dir_name, 'clean_lagos_restaurants.csv')

restaurants_init = pd.read_csv(location)

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


# removing stopwords in the review text
stop_words = stopwords.words('english')
restaurants_init['review_text'] = restaurants_init['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

# removing stopwords in review title
restaurants_init['review_title'] = restaurants_init['review_title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

# removing punctuation
punctuation_and_symbols = r'\[^\w\s\]'
restaurants_init['review_text'] = restaurants_init['review_text'].str.replace(punctuation_and_symbols, ' ', regex=True)
restaurants_init['review_title'] = restaurants_init['review_title'].str.replace(punctuation_and_symbols, ' ', regex=True)

restaurants_init['review_text'] = restaurants_init['review_text'].str.replace("  ", " ")
restaurants_init['review_title'] = restaurants_init['review_title'].str.replace("  ", " ")

# lemmatizing
import corpora
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
st.write("For restaurants with different locations, kindly add it, i.e. VI, Lekki, Ikeja")

st.write("Check out our [hotel analyser](https://lag-hotel.streamlit.app/)")

form = st.form(key='sentiment-form')
user_input = form.text_area("Enter a restaurant's name")
submit = form.form_submit_button('Submit')

matched_restaurant_names = [x for x in result['restaurant_name'].values if user_input.lower() in x.lower()]
condition = len(matched_restaurant_names) > 0

if submit:
  if not user_input or not user_input.strip():
    st.error("The restaurant name field is required")
  elif len(user_input) == 1:
    st.error("Please type out full restaurant name")
  else: 
    if condition == True:
      for restaurant_name in matched_restaurant_names:
        whole_row = result[result['restaurant_name'] == restaurant_name]
        score = whole_row['opinion'].values[0]
        if score == 'Positive':
          st.success(f'Many customers find {restaurant_name} a good place to spend their money!')
        elif score == 'Negative' or score == 'Neutral':
          st.success(f'The average customer finds {restaurant_name} not so great for eating out. Maybe try somewhere else?')
    else:
        st.error(f'{user_input} is not in our database, we apologize about that.')
