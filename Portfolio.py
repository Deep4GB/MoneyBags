import yfinance as yf

def get_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None

def calculate_portfolio_allocation(ticker_symbols, start_date, end_date, total_investment):
    stock_data = {}
    for symbol in ticker_symbols:
        data = get_stock_data(symbol, start_date, end_date)
        if data is not None:
            stock_data[symbol] = data
        else:
            return None
    
    total_closing_price = sum(data['Close'].iloc[-1] for data in stock_data.values())
    allocation_percentage = {symbol: data['Close'].iloc[-1] / total_closing_price for symbol, data in stock_data.items()}
    allocation_dollars = {symbol: percentage * total_investment for symbol, percentage in allocation_percentage.items()}
    
    return allocation_percentage, allocation_dollars

def main():
    ticker_symbols = input("Enter stock ticker symbols (separated by comma): ").strip().upper().split(',')
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    total_investment = float(input("Enter total investment amount: $"))

    allocation_percentage, allocation_dollars = calculate_portfolio_allocation(ticker_symbols, start_date, end_date, total_investment)

    if allocation_percentage is not None:
        print("\nOptimal Portfolio Allocation:")
        for symbol, percentage in allocation_percentage.items():
            print(f"{symbol}: {percentage:.2%} (Amount: ${allocation_dollars[symbol]:.2f})")
    else:
        print("Error: Unable to calculate portfolio allocation.")

if __name__ == "__main__":
    main()
