<h2 align="center"> ━━━━━━  ❖  ━━━━━━ </h2>
<h1 align="center"> MoneyBags: Stock Market Prediction Application </h1>

MoneyBags is a stock market prediction application developed using Python and Flask. This application aims to provide users with predictions of stock prices based on historical data using machine learning algorithms, along with sentiment analysis of news articles related to the stock.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Installation](#installation)
- [Contributions](#contributions)
- [License](#license)
- [Disclaimer](#disclaimer)
- [Team Members](#team-members)

## Features

1. **Stock Prediction**: Users can input a stock ticker symbol and select a date range to train the prediction model. The application then utilizes machine learning algorithms to predict the stock prices for the next week.

2. **Sentiment Analysis**: MoneyBags performs sentiment analysis on news articles related to the stock, providing insights into the overall sentiment (positive, negative, or neutral) surrounding the stock.

3. **Multiple Algorithms**: MoneyBags incorporates multiple machine learning algorithms such as LSTM and Random Forest regression for stock price prediction. These algorithms are trained on historical stock data to provide accurate predictions.

4. **Stacking**: The application employs stacking techniques to combine the predictions from multiple algorithms, aiming to improve the overall accuracy and reliability of the predictions.

5. **User-friendly Interface**: MoneyBags provides a simple and intuitive interface for users to input the stock symbol and select the training date range. The predicted stock prices and sentiment analysis results are displayed in an easy-to-understand format.

## Usage

1. **Input Stock Symbol**: Enter the ticker symbol of the stock you want to predict.

2. **Select Date Range**: Choose the start and end dates for training the prediction model.

3. **Get Predictions**: Click the "Predict" button to generate predictions for the next week's stock prices.

4. **View Results**: The application displays the predicted stock prices for each day of the upcoming week, along with sentiment analysis results, allowing users to make informed investment decisions.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Deep4GB/MoneyBags.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```
   python app.py
   ```

4. Access the application in your web browser at `http://localhost:5000`.

## Contributions

Contributions to MoneyBags are welcome! If you have any suggestions for improvement or would like to contribute new features, feel free to submit a pull request.

## License

MoneyBags is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Disclaimer

MoneyBags is for educational and informational purposes only. The predictions provided by the application are based on historical data and machine learning algorithms, and there is no guarantee of accuracy or profitability. Users should conduct their own research and consult with financial professionals before making investment decisions.

## Team Members

- [Cameron Bussom](https://github.com/)
- [Dev Patel](https://github.com/)
- [Deep Patel](https://github.com/Deep4GB)
- [Darsh Patel](https://github.com/)
