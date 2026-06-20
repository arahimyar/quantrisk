from backtester import Backtest
from plotter import Plotter
import os
import numpy as np
import time
import pandas as pd

alpha = 0.01
distributions = ["Gaussian", "StudentT", "NIG"]
results_collector = {dist: [] for dist in distributions}


path = "/Users/hasibrahimyar/Desktop/Projects/ARMAGARCH/DATA"
all_files = os.listdir(path)
csv_files = [f for f in all_files if f.endswith(".csv")]
print(f"Found {len(csv_files)} tickers in the '{path}' directory.\n")

for file_name in csv_files:
    ticker, _ = os.path.splitext(file_name)
    file_path = os.path.join(path, file_name)
    try:
        df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
        df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        prices = df["Close"].dropna()
        returns = np.log(prices / prices.shift(1)).dropna().values
        if len(returns) < 1000:
            print(f"Skipping {ticker}: Insufficient history ({len(returns)} days).")
            continue
            
        print(f"Running backtest for {ticker}...")
        
        returns = returns[-1000:]
        for dist in distributions:
            start_time = time.time()
            test = Backtest(returns, alpha, dist)
            test.rolling_window(250)
            end_time = time.time()
            elapsed_time = end_time - start_time
            stat_results = test.statistical_tests()
            row_data = {"Asset": ticker}
            row_data.update(stat_results)
            row_data["Elapsed_Time_Sec"] = round(elapsed_time, 2)
            results_collector[dist].append(row_data)
            print(dist, stat_results)

        print(f"SUCCESS: {ticker} processed cleanly.")
        
    except Exception as e:
        print(f"Failed to process {ticker}. Error: {e}")
        continue


for dist in distributions:
    dist_df = pd.DataFrame(results_collector[dist])
    
    if not dist_df.empty:
        dist_df.set_index("Asset", inplace=True)
        output_filename = f"{path}/results_{dist}.csv"
        dist_df.to_csv(output_filename)
        print(f"  Saved: {output_filename}")
    else:
        print(f"  Warning: No data collected for distribution '{dist}', skipping file export.")
