# first line: 19
@memory.cache
def analyze_sentiment(restaurants_init):
  #Preprocessing
  # removing manager feedback
  feedback_string = 'Thank you for'
  manager = restaurants_init.loc[restaurants_init['review_text'].str.contains(feedback_string)]
  restaurants_init = restaurants_init.drop(manager.index.values, axis=0).drop(columns='Unnamed: 0')

  # converting all text to lowercase
  restaurants_init['review_text'] = restaurants_init['review_text'].str.lower()
  restaurants_init['review_title'] = restaurants_init['review_title'].str.lower()

  # converting emojis to words
  # Converting emojis to words
  def convert_emojis(text):
    for emot in UNICODE_EMOJI:
      text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
    return text

  # Passing both functions to the review text and review title
  restaurants_init['review_text'] = restaurants_init['review_text'].apply(convert_emojis)
  restaurants_init['review_title'] = restaurants_init['review_title'].apply(convert_emojis)

  # removing stopwords in the review text and title
  stop_words = stopwords.words('english')
  restaurants_init['review_text'] = restaurants_init['review_text'].apply(lambda x: " ".\
    join(x for x in x.split() if x not in stop_words))
  restaurants_init['review_title'] = restaurants_init['review_title'].apply(lambda x: " ".\
    join(x for x in x.split() if x not in stop_words))

  # removing punctuation
  punctuation_and_symbols = r'\[^\w\s\]'
  restaurants_init['review_text'] = restaurants_init['review_text'].\
  str.replace(punctuation_and_symbols, ' ', regex=True).str.replace("  ", " ")
  restaurants_init['review_title'] = restaurants_init['review_title'].\
    str.replace(punctuation_and_symbols, ' ', regex=True).str.replace("  ", " ")

  # lemmatizing
  restaurants_init['review_text'] = restaurants_init['review_text'].apply(lambda x: " ".\
    join([Word(word).lemmatize() for word in x.split()]))
  restaurants_init['review_title'] = restaurants_init['review_title'].apply(lambda x: " ".\
    join([Word(word).lemmatize() for word in x.split()]))

  # sentiment analysis
  analyzer = SentimentIntensityAnalyzer()
  sentiments = [analyzer.polarity_scores(row) for row in restaurants_init['review_text']]
  df_sentiments = pd.DataFrame(sentiments)

  # joining both dataframes
  restaurants_final = pd.concat([restaurants_init.reset_index(drop=True), df_sentiments], axis=1)
  # choosing sentiment based on compound score
  restaurants_final['Sentiment'] = np.where(restaurants_final['compound'] > 0.05, 'Positive',
         (np.where(restaurants_final['compound'] < -0.05, 'Negative', 'Neutral')))
  restaurants_final = restaurants_final.drop(['neg', 'neu', 'pos', 'compound'], axis=1)

  result = restaurants_final.groupby('restaurant_name')['Sentiment'].value_counts().unstack().fillna(0)
  result['opinion'] = result[['Negative','Neutral', 'Positive']].idxmax(axis=1)

  return result.reset_index()
