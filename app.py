from textblob import TextBlob
import requests
from flask import Flask, jsonify, render_template, request
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)


# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning)


# Function to fetch historical stock data from Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()


# Function to calculate Moving Average (MA) for a given window
def calculate_moving_average(data, window):
    return data['Close'].rolling(window=window).mean()


# Function to add Moving Average as a feature to the dataset
def add_moving_average_feature(data, window):
    data['MA'] = calculate_moving_average(data, window)
    return data


# Function to preprocess data for training the model
def preprocess_data(data, ma_window):
    data = add_moving_average_feature(data, ma_window)
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(
        data['Date']).dt.date.apply(lambda x: x.toordinal())

    # Explicitly include the 'Close' column
    data['Close'] = data['Close']

    # Fill NaN values with 0
    data = data.fillna(0)

    return data


# Function to train a Random Forest Regressor model
def train_model(data):
    X = data[['Date', 'Open', 'High', 'Low', 'Volume', 'MA']]
    y = data['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, scaler


# Function to predict the next day's closing price
def predict_price(model, last_date, scaler, open_price, high_price, low_price, volume):
    next_date = last_date + timedelta(days=1)
    next_date_ordinal = next_date.toordinal()
    # 0 is a placeholder for MA since it's not available for the next day
    input_data = np.array(
        [[next_date_ordinal, open_price, high_price, low_price, volume, 0]])
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)[0]


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stock')
def stock():
    return render_template('stock.html')


@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')


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


@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')


@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_route():
    symbol = request.form['tickerSymbol'].upper()
    news_articles = fetch_news(symbol)
    if news_articles:
        average_sentiment = analyze_sentiment(news_articles)
        sentiment_results = {
            'symbol': symbol,
            'average_sentiment': average_sentiment,
            'news_articles': news_articles
        }
        return render_template('sentiment.html', sentiment_results=sentiment_results)
    else:
        return render_template('sentiment.html', message="No news articles found for the provided symbol.")


@app.route('/get_stock_price', methods=['POST'])
def get_stock_price():
    ticker_symbol = request.form['tickerSymbol']
    stock = yf.Ticker(ticker_symbol)

    try:
        price_data = stock.history(period='1d')
        if not price_data.empty:
            # Get the latest closing price
            current_price = price_data['Close'].iloc[-1]
            return {'currentPrice': str(current_price)}
        else:
            return {'currentPrice': 'Failed to fetch current price'}
    except Exception as e:
        print(f"Error fetching stock price: {e}")
        return {'currentPrice': 'Failed to fetch current price'}


@app.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')


@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    prediction_text = ""  # Define an initial value for prediction_text
    current_price = ""    # Define an initial value for current_price
    chart_data = None     # Define an initial value for chart_data

    if request.method == 'POST':
        # Handle form submission here
        # Fetch form data
        symbol = request.form['symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        ma_window = request.form['ma_window']

        # Fetch historical stock data
        stock_data = get_stock_data(symbol, start_date, end_date)

        if stock_data.empty:
            prediction_text = "No data found for the given symbol and date range"
        else:
            # Preprocess data
            processed_data = preprocess_data(stock_data, int(ma_window))

            if len(processed_data) > 0:  # Check if processed_data contains samples
                try:
                    # Train the LSTM model
                    lstm_model, X_test_lstm, y_test_lstm, scaler_lstm = train_model(processed_data)

                    # Train the Random Forest model
                    rf_model, X_test_rf, y_test_rf, scaler_rf = train_model(processed_data)

                    # Get the latest data for prediction
                    latest_data = processed_data.iloc[-1]
                    open_price = latest_data['Open']
                    high_price = latest_data['High']
                    low_price = latest_data['Low']
                    volume = latest_data['Volume']

                    # Fetch current or last closing price
                    stock = yf.Ticker(symbol)
                    price_data = stock.history(period='1d')
                    if not price_data.empty:
                        current_price = price_data['Close'].iloc[-1]

                    # Predict using LSTM
                    lstm_next_price = predict_price(
                        lstm_model, latest_data.name, scaler_lstm, open_price, high_price, low_price, volume)

                    # Predict using Random Forest
                    rf_next_price = predict_price(
                        rf_model, latest_data.name, scaler_rf, open_price, high_price, low_price, volume)

                    # Stack predictions
                    stacked_price = (lstm_next_price + rf_next_price) / 2

                    # Update prediction_text with the stacked price
                    prediction_text = f"Stacked predicted closing price: ${stacked_price:.2f}"

                    chart_data = {"x": [1, 2, 3], "y": [10, 20, 30]}

                    # Evaluate models
                    # Evaluate LSTM model
                    lstm_predictions = lstm_model.predict(X_test_lstm)
                    lstm_mae = mean_absolute_error(y_test_lstm, lstm_predictions)
                    lstm_mse = mean_squared_error(y_test_lstm, lstm_predictions)
                    lstm_r2 = r2_score(y_test_lstm, lstm_predictions)

                    print("LSTM Model Evaluation:")
                    print(f"Mean Absolute Error (MAE): {lstm_mae:.2f}")
                    print(f"Mean Squared Error (MSE): {lstm_mse:.2f}")
                    print(f"R-squared (R^2) Score: {lstm_r2:.2f}")

                    # Evaluate Random Forest model
                    rf_predictions = rf_model.predict(X_test_rf)
                    rf_mae = mean_absolute_error(y_test_rf, rf_predictions)
                    rf_mse = mean_squared_error(y_test_rf, rf_predictions)
                    rf_r2 = r2_score(y_test_rf, rf_predictions)

                    print("\nRandom Forest Model Evaluation:")
                    print(f"Mean Absolute Error (MAE): {rf_mae:.2f}")
                    print(f"Mean Squared Error (MSE): {rf_mse:.2f}")
                    print(f"R-squared (R^2) Score: {rf_r2:.2f}")

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    prediction_text = "Error occurred during prediction"
            else:
                prediction_text = "Insufficient data for prediction"
                symbol = request.args.get('symbol')
                # Fetch current or last closing price
                stock = yf.Ticker(symbol)
                price_data = stock.history(period='1d')
                if not price_data.empty:
                    current_price = price_data['Close'].iloc[-1]
                    current_price = round(current_price, 2)
                    current_price = "{:.2f}".format(current_price)

    return render_template('prediction.html', symbol=symbol, current_price=current_price, prediction_text=prediction_text, chart_data=chart_data)


if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5100, debug=True)
