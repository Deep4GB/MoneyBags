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

### Stock Prediction
- **Description**: The stock prediction feature utilizes advanced machine learning techniques including Long Short-Term Memory (LSTM), Random Forest Regression, and stacking to forecast stock prices accurately. It incorporates historical stock data, moving averages, and various technical indicators to train the prediction models.
- **Functionality**:
  - Users can input a stock ticker symbol and select a date range for training the prediction model.
  - The application preprocesses the historical stock data, calculates moving averages, and extracts relevant features.
  - Multiple machine learning algorithms, including LSTM and Random Forest Regression, are trained on the preprocessed data to predict future stock prices.
  - Stacking techniques are employed to combine the predictions from different algorithms, enhancing the overall accuracy and reliability of the forecasts.
  - Predicted stock prices for the next week are generated based on the trained models and displayed to the users.
- **Benefits**:
  - Provides users with valuable insights into potential price movements of stocks, aiding in informed decision-making for trading and investment strategies.
  - Incorporates advanced machine learning algorithms to analyze complex patterns in historical stock data, resulting in more accurate predictions.
  - Employs stacking techniques to leverage the strengths of multiple models, leading to improved forecasting performance.

### Sentiment Analysis
- **Description**: The sentiment analysis feature enables users to gauge the overall sentiment surrounding a particular stock by analyzing news articles related to the stock. It utilizes the Polygon API to fetch relevant news articles and calculates sentiment scores to assess whether the sentiment is positive, negative, or neutral.
- **Functionality**:
  - Users input a stock ticker symbol for which they want to analyze sentiment.
  - The application fetches news articles related to the specified stock using the Polygon API.
  - Sentiment analysis is performed on the retrieved articles, considering both the titles and descriptions.
  - The sentiment scores are aggregated and analyzed to determine the overall sentiment surrounding the stock.
  - Users are provided with insights into the sentiment analysis results, along with a summary of the news articles affecting the stock sentiment.
- **Benefits**:
  - Offers users valuable insights into market sentiment, helping them understand the broader market perception of a particular stock.
  - Provides a comprehensive view of sentiment trends, allowing users to make more informed decisions regarding their investments.
  - Empowers users to stay updated with the latest news and sentiment dynamics, facilitating proactive decision-making in response to market sentiment changes.

### Portfolio Analysis and Optimization
- **Description**: The portfolio analysis and optimization feature enable users to optimize their investment portfolios by allocating funds across multiple stocks based on historical performance, risk tolerance, and investment objectives. Users can input multiple ticker symbols and the desired investment amount for each stock to receive insights on the optimal allocation strategy.
- **Functionality**:
  - Users input multiple stock ticker symbols along with the desired investment amount for each stock.
  - The application analyzes historical performance data for the selected stocks and calculates risk-adjusted returns, correlations, and other portfolio metrics.
  - Based on the inputted data and user-defined constraints, the application determines the optimal allocation of funds across the selected stocks to maximize returns while minimizing risk.
  - Users receive insights into diversification strategies, rebalancing recommendations, and portfolio performance metrics to make informed investment decisions.
- **Benefits**:
  - Helps users construct well-diversified portfolios tailored to their investment objectives and risk tolerance.
  - Provides personalized investment recommendations based on quantitative analysis and optimization techniques.
  - Empowers users to optimize their investment portfolios for better risk-adjusted returns and long-term wealth accumulation.


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

4. Access the application in your web browser at `http://localhost:5100`.

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
