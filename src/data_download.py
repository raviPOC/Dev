import yfinance as yf
import pandas as pd
import os
import argparse

def download_data(ticker, start_date, end_date, output_dir="data"):
    """
    Downloads stock data from Yahoo Finance and saves it to a CSV file.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print(f"No data found for {ticker}.")
        return

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Reset index to make Date a column
    data.reset_index(inplace=True)
    
    file_path = os.path.join(output_dir, f"{ticker}_{start_date}_{end_date}.csv")
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data using yfinance")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    download_data(args.ticker, args.start, args.end, args.output)
