from nltk.corpus import stopwords
from textblob import Word
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from emot.emo_unicode import UNICODE_EMOJI
from joblib import Memory

import os
import nltk
import pandas as pd
import streamlit as st

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

cachedir = './cachedir'
memory = Memory(location=cachedir, verbose=0)
@memory.cache
def analyze_sentiment(hotels):
    # Preprocessing
    feedback_string = 'Thank you for'
    manager = hotels.loc[hotels['review_text'].str.contains(feedback_string)]
    hotels = hotels.drop(manager.index.values, axis=0).drop(columns='Unnamed: 0')

    hotels['review_text'] = hotels['review_text'].str.lower()
    hotels['review_title'] = hotels['review_title'].str.lower()

    def convert_emojis(text):
        for emot in UNICODE_EMOJI:
            text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()))
        return text

    hotels['review_text'] = hotels['review_text'].apply(convert_emojis)
    hotels['review_title'] = hotels['review_title'].apply(convert_emojis)

    stop_words = set(stopwords.words('english'))
    hotels['review_text'] = hotels['review_text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    hotels['review_title'] = hotels['review_title'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    punctuation_and_symbols = r'[^\w\s]'
    hotels['review_text'] = hotels['review_text'].str.replace(punctuation_and_symbols, ' ', regex=True).str.replace(
        '  ', ' ')
    hotels['review_title'] = hotels['review_title'].str.replace(punctuation_and_symbols, ' ', regex=True).str.replace(
        '  ', ' ')

    hotels['review_text'] = hotels['review_text'].apply(
        lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))
    hotels['review_title'] = hotels['review_title'].apply(
        lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))

    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(row) for row in hotels['review_text']]
    df_sentiments = pd.DataFrame(sentiments)

    all_hotels = pd.concat([hotels.reset_index(drop=True), df_sentiments], axis=1)
    all_hotels['Sentiment'] = all_hotels['compound'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
    result = all_hotels.groupby('hotel_name')['Sentiment'].value_counts().unstack().fillna(0)
    result['opinion'] = result[['Negative', 'Neutral', 'Positive']].idxmax(axis=1)
    return result.reset_index()

# Load CSV
dir_name = os.path.abspath(os.path.dirname(__file__))
location = os.path.join(dir_name, 'clean_lagos_hotels.csv')
hotels = pd.read_csv(location)

st.title("Lagos Hotel's Sentiment Analyser App")
st.write("Get an accurate feel of what people think about a hotel's service!")
st.write("For hotel's with different locations, kindly add it, i.e. VI, Lekki, Ikeja")

st.write("Check out our [restaurant analyser](https://lag-rest.streamlit.app/)")

form = st.form(key='sentiment-form')
user_input = form.text_area("Kindly enter a hotels name")
submit = form.form_submit_button('Submit')

# Check if the user-input hotel exists
matched_hotel_names = [x for x in set(hotels['hotel_name'].values) if isinstance(x, str) and user_input.lower() in x.lower()]
doesUserHotelExist = len(matched_hotel_names) > 0

if submit:
    if not user_input or not user_input.strip():
        st.error("The hotel name field is required")
    elif len(user_input) == 1:
        st.error("Please type out full restaurant name")
    else:    
        if not doesUserHotelExist:
            st.error(f'{user_input} is not in our database, we apologize about that.')
        else:
            # Display results
            result = analyze_sentiment(hotels)
            for hotel_name in matched_hotel_names:
                whole_row = result[result['hotel_name'] == hotel_name]
                score = whole_row['opinion'].values[0]
                if score == 'Positive':
                    st.success(f'Many customers find {hotel_name} a good place to spend their money!')
                elif score == 'Negative' or score == 'Neutral':
                    st.success(
                        f'The average customer finds {hotel_name} not so great for staying in. Maybe try somewhere else?')