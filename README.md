# ANALYSING REVIEWS OF HOTELS AND RESTAURANTS IN NIGERIA TO DETERMINE CUSTOMER SENTIMENT

## Project description
Choosing hotels and restaurants is a time-consuming process, often involving weeks of deliberation. To make informed decisions, prospective guests increasingly rely on online reviews, which sometimes contain huge volume of comments - some of which may be manipulated. Our goal is to empower Nigerians in making better informed choices and help restaurants and hotels improve their services by leveraging guest feedback. This study's potential extends to diverse areas within the hospitality industry, such as tourism, travel, recreation, and entertainment.

Check the deployed restaurant sentiment analyzer model [**here**](https://lag-rest.streamlit.app/).

## Table of Contents

## Getting Started

Step-by-step instructions on how to install and set up the project

- Fork the repo and download to your system.
- In your working directory, run `pip install -r requirements.txt` in your terminal
- Run `streamlit run restaurant-model-building.py`

### Resources that inspired the project
- Singh, H. (2006) The importance of customer satisfaction in relation to customer loyalty ..., The Importance of Customer Satisfaction in Relation to Customer Loyalty and  Retention. Available at: https://www.van-haaften.nl/images/documents/pdf/The%20Importance%20of%20Customer%20Satisfaction%20in%20Relation%20to%20Customer%20Loyalty%20and%20Retention.pdf (Accessed: 15 October 2023).
- Nowak, A. (2022) The history of online reviews, Expert Reputation. Available at: https://expertreputation.com/the-history-of-online-reviews/#:~:text=The%20First%20Online%20Reviews,by%20Yellow%20Pages%20in%202001. (Accessed: 15 October 2023).
- Editor (2021) Sentiment analysis in Hotel Reviews: Developing a decision-making assistant for Travelers, AltexSoft. Available at: https://www.altexsoft.com/blog/sentiment-analysis-hotel-reviews/ (Accessed: 15 October 2023).
- Nwakanma, Cosmas & A.C, Ogbonna & Etus, Chukwuemeka & Nwifor, Emmanuella & Onyebuchi, Jennifer & Ugwueke, Esther. (2019). Predictive Analytics of Customer Sentiments towards Nigerian Hospitality Industry: Case study Approach.
- Nigerians spent n4.6 trillion eating out in 2019, says NBS (2020) TheCable. Available at: https://www.thecable.ng/nigerians-spent-n4-6-trillion-eating-out-in-2019-says-nbs (Accessed: 15 October 2023).
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

## Requirements:
- **Data Sourcing:** Data was scraped from <a href="https://www.tripadvisor.com/">TripAdvisor</a> - mainly restaurants and hotels in Lagos, Nigeria.
- **Data Cleaning and Prep**: Data was labelled and cleaned using Excel and Pandas to remove noise or meaningless data that could affect the accuracy of the model.
- **Modeling**: The data underwent classification through the application of sentiment analysis with the VADER (Valence Aware Dictionary and sEntiment Reasoner) package, which is a sentiment analysis tool tuned to social media sentiment.
- **Model Deployment :** The model would be deployed using a web app for use by everyone particularly Nigerians.
- **Requirements.txt**: A file for all dependecies required. There's one in the Restaurants folder and another in the Hotels folder.

## Challenges faced:
- Scraping reviews from Google. There are more restaurants reviewed with Google than TripAdvisor, since TripAdvisor needs creating an account. This limited our scope to just TripAdvisor.
- VADER sentiment does not accurately detect sarcasm in Nigerian English as some of the reviews came off as sarcastic but were rated positive.
- Deploying with Streamlit. Worked fine on local machine and then gave many issues during deployment. 

## Potential work:
- Scrapping reviews from more sites. This will help give a better accurate overall sentiment.
- Training our model in the Nigerian English context.

## Acknowledgments
### Contributors
- Daniel Otulagun
- Sarah Akinkunmi
- Olatunde Ogunboyejo

### Mentor
- Orevaoghene Ahia


## Contact
- <a href="https://www.linkedin.com/in/otulagun/">Daniel Otulagun </a>
- <a href="https://www.linkedin.com/in/sarah-akinkunmi/">Sarah Akinkunmi </a>
- <a href="https://www.linkedin.com/in/ogunboyejo-olatunde/">Olatunde Ogunboyejo </a>
- <a href="https://www.linkedin.com/in/orevaoghene-ahia/">Orevaoghene Ahia </a> 


