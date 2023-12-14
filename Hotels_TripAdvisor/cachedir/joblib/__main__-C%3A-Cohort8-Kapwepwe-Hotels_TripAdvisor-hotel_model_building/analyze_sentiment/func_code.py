# first line: 17
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
