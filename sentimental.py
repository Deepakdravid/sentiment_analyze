import string
import random
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import re

df = pd.read_csv('/home/dravid/Downloads/jnotebook/test.csv')


df = df.rename(columns={'tweet': 'text'})

def clean_text(text):
   
    text = re.sub(r'[^\w\s]|_+', '', text)
    text = text.lower()
    text = re.sub('\s+', ' ', text).strip()
    return text



def analyze_sentiment(text):
    cleaned_text = clean_text(text)
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    if sentiment_score > 0.0:
        emotion = "positive"
        category = random.choice(['happy','interaction','personal status','sentiment','whether','general'])
    elif sentiment_score < -0.0:
        emotion = "negative"
        category = random.choice(['sad','unhappy','disopointment','dislike','discomfort','depressed'])
    else:
        emotion = "neutral"
        category = random.choice(['sports','casual','personal update','expressing indifferent'])
    
    return sentiment_score, emotion ,category
for comment in df['text'][:10]:  
     sentiment_score, emotion , category = analyze_sentiment(comment)
     print(f"Comment: '{comment}' - Score: {sentiment_score} - Emotion: {emotion} -category: {category}")



df['polarity'], df['sentiment'], df['category'] = zip(*df['text'].apply(analyze_sentiment))


plt.hist(df['polarity'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.title('Sentiment Polarity Distribution')
plt.grid(True)
plt.show()


sentiment_counts = df['sentiment'].value_counts()
plt.bar(sentiment_counts.index, sentiment_counts.values, color='lightgreen')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.title('Sentiment Distribution')
plt.grid(True)
plt.show()
df.head()
print(df.head)
