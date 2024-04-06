import requests
from textblob import TextBlob

def fetch_news(symbol):
    api_key = '0W4bW1G55N6q_a4lEauU960Buqr9WTQy'
    url = f'https://api.polygon.io/v2/reference/news?ticker={symbol}&apiKey={api_key}'
    
    try:
        response = requests.get(url)
        news_data = response.json()['results']
        return news_data
    except Exception as e:
        print(f"Error fetching news data: {e}")
        return []

def analyze_sentiment(news_articles):
    sentiment_scores = []
    
    for article in news_articles:
        title = article['title']
        description = article['description']
        
        # Concatenate title and description for sentiment analysis
        text = f"{title}. {description}"
        
        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        sentiment_scores.append(sentiment_score)
    
    # Calculate average sentiment score
    if sentiment_scores:
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    else:
        average_sentiment = 0.0
    
    return average_sentiment

def main():
    symbol = input("Enter the ticker symbol of the stock you want to analyze: ").upper()
    
    news_articles = fetch_news(symbol)
    
    if news_articles:
        average_sentiment = analyze_sentiment(news_articles)
        print(f"Average sentiment score for {symbol}: {average_sentiment}")
        
        if average_sentiment > 0.5:
            print("The sentiment is strongly positive.")
        elif average_sentiment > 0:
            print("The sentiment is slightly positive.")
        elif average_sentiment < -0.5:
            print("The sentiment is strongly negative.")
        elif average_sentiment < 0:
            print("The sentiment is slightly negative.")
        else:
            print("The sentiment is neutral.")
        
        print("Articles contributing to sentiment:")
        for article in news_articles:
            title = article['title']
            description = article['description']
            sentiment_score = TextBlob(f"{title}. {description}").sentiment.polarity
            
            if (sentiment_score > 0.5 and average_sentiment > 0.5) or (sentiment_score < -0.5 and average_sentiment < -0.5):
                print(f"- {title} (Strongly {('Positive' if sentiment_score > 0 else 'Negative')})")
            elif (sentiment_score > 0 and average_sentiment > 0) or (sentiment_score < 0 and average_sentiment < 0):
                print(f"- {title} (Slightly {('Positive' if sentiment_score > 0 else 'Negative')})")
        
    else:
        print("No news articles found for the provided symbol.")

if __name__ == "__main__":
    main()
