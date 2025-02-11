import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
from collections import Counter
from datetime import timedelta
import requests

from wordcloud import WordCloud
import nltk
import streamlit as st

# Download required NLTK data
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Memory Analysis",
    page_icon="❤️",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* This targets all paragraph text within the main container */
    .reportview-container .main .block-container p {
        font-size: 30px;
    }
    /* Optionally, increase font size for any text inside st.write() or st.markdown() that isn’t a heading */
    .stMarkdown, .block-container {
        font-size: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Add a title and a cute message
st.title("Chat Analysis Dashboard")
st.markdown("### Hi Princess, Welcome to our little memory lane! 💖")
st.write("Here's a look at our chat moments and the story of our relationship, all in one place. I wish that these conversations never stop.")

#################################
# 1. READ & PARSE THE CHAT FILE #
#################################
pattern = (
    r'(\d{1,2}/\d{1,2}/\d{4}),\s*'  # Date (e.g., "04/10/2023,")
    r'(\d{1,2}:\d{2})'              # Time (e.g., "7:58")
    r'(\s*[apAP][mM])?'             # Optional am/pm (with possible spaces)
    r'\s*-\s+'                     # Separator " - "
    r'(.*?):\s+'                   # Sender (up to the colon+space)
    r'(.*)'                        # Message text
)

CHAT_FILE_URL = 'https://drive.google.com/uc?export=download&id=1KJSx7XL2f0Odu0_p5Fsu1n-BAGn65dvN'
try:
    response = requests.get(CHAT_FILE_URL)
    response.raise_for_status()
    chat_data = response.text
    st.write("Chat file loaded successfully!")
except Exception as e:
    st.error("Error loading chat file: " + str(e))
    st.stop()
    
messages = re.findall(pattern, chat_data)
st.write(f"Found {len(messages)} messages in chat data.")

df = pd.DataFrame(messages, columns=['date', 'time', 'ampm', 'sender', 'text'])
df['datetime_str'] = df['date'] + ' ' + df['time'] + df['ampm'].fillna('')
df['datetime'] = pd.to_datetime(df['datetime_str'], format='%d/%m/%Y %I:%M %p', errors='coerce')
df.dropna(subset=['datetime'], inplace=True)
df.sort_values('datetime', inplace=True)
df.reset_index(drop=True, inplace=True)
df['date_only'] = df['datetime'].dt.date

#################################
# 2. SENTIMENT TIMELINE (MOOD GRAPH)
#################################
analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
sentiment_by_day = df.groupby('date_only')['sentiment'].mean().reset_index()

st.markdown("## Sentiment Timeline (Mood Graph)")
st.markdown("""
**Compound Score Explanation:**  
The compound score is a number between -1 and 1, where:
- **-1** indicates extremely negative sentiment,
- **0** indicates neutral sentiment,
- **1** indicates extremely positive sentiment.
""")

# Create the sentiment chart figure
fig_sentiment, ax_sentiment = plt.subplots(figsize=(12, 6))
sns.lineplot(data=sentiment_by_day, x='date_only', y='sentiment', marker='o', ax=ax_sentiment)
ax_sentiment.set_title("Average Daily Sentiment")
ax_sentiment.set_xlabel("Date")
ax_sentiment.set_ylabel("Average Sentiment (Compound Score)")
plt.xticks(rotation=45)
plt.tight_layout()

# (Removed the first st.pyplot(fig_sentiment) call)

#################################
# 3. CHAT ACTIVITY HEATMAP
#################################
df['weekday'] = df['datetime'].dt.day_name()
df['hour'] = df['datetime'].dt.hour
activity = df.groupby(['weekday', 'hour']).size().reset_index(name='count')
ordered_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
activity['weekday'] = pd.Categorical(activity['weekday'], categories=ordered_weekdays, ordered=True)
activity = activity.sort_values(['weekday', 'hour'])
heatmap_data = activity.pivot(index="weekday", columns="hour", values="count")

fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="g", ax=ax_heatmap)
ax_heatmap.set_title("Chat Activity Heatmap (by Weekday and Hour)")
ax_heatmap.set_xlabel("Hour of Day")
ax_heatmap.set_ylabel("Weekday")

#################################
# 4. LONGEST CHAT STREAK
#################################
unique_dates = sorted(set(df['date_only']))
max_streak = 0
current_streak = 1
max_streak_start = unique_dates[0]
max_streak_end = unique_dates[0]

for i in range(1, len(unique_dates)):
    if (unique_dates[i] - unique_dates[i - 1]).days == 1:
        current_streak += 1
        if current_streak > max_streak:
            max_streak = current_streak
            max_streak_end = unique_dates[i]
            max_streak_start = unique_dates[i - current_streak + 1]
    else:
        current_streak = 1

longest_streak_text = f"Longest chat streak: {max_streak} days, from {max_streak_start} to {max_streak_end}."

#################################
# 5. BIGRAM WORD CLOUD
#################################
stop_words = set(stopwords.words('english'))
custom_stopwords = {
     'ok', 'nhi', 'mujhe', 'haan', 'na', 'hmm', 
    'bhai', '<media', 'omitted>', 'de', 'kr','null','ho','h','hi','kr','hai','deleted_message','message_edited','deleted_message'
}
stop_words.update(custom_stopwords)

def clean_message(msg):
    msg = msg.lower()
    msg = msg.translate(str.maketrans('', '', string.punctuation))
    tokens = msg.split()
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

def generate_ngrams(tokens, n=2):
    return zip(*[tokens[i:] for i in range(n)])

all_bigrams = []
for text in df['text']:
    if text.strip().lower() == '<media omitted>':
        continue
    tokens = clean_message(text)
    bigram_tuples = generate_ngrams(tokens, n=2)
    bigram_strs = ["_".join(bigram) for bigram in bigram_tuples]
    all_bigrams.extend(bigram_strs)

if all_bigrams:
    bigrams_for_cloud = " ".join(all_bigrams)
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(bigrams_for_cloud)
    fig_wordcloud, ax_wordcloud = plt.subplots(figsize=(10, 5))
    ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
    ax_wordcloud.axis('off')
else:
    fig_wordcloud = None

#################################
# STREAMLIT DASHBOARD LAYOUT
#################################
# Display everything here in the final layout
st.header("Sentiment Timeline (Mood Graph)")
st.pyplot(fig_sentiment)

st.header("Chat Activity Heatmap")
st.pyplot(fig_heatmap)

st.header("Our Longest Chat Streak")
st.write(longest_streak_text)

st.header("Our Fav Words")
if fig_wordcloud:
    st.pyplot(fig_wordcloud)
else:
    st.write("No bigrams found to display a word cloud.")
