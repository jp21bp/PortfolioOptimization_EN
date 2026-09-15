"""
This file details the data collection process using YFinance APIs
"""


##### Importing libraries
import yfinance as yf
import os

##### Directory path
data_path = os.path.join(
    os.getcwd(),
)

#######################################################################
    # Top 40 Inside SPLAC #
##### Tickers
with open(f'{data_path}/SPLAC_tickers.txt', 'r', encoding='utf-8') as file:
    tickers = file.readlines()


##### Download and Collection
start_date = "2000-01-01"
end_date = "2026-08-31"
    #Top 40 recorded on 8/31/26
df_raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)

##### Filter to only save 'Adj Close' and dates
df_raw_AC = df_raw.xs('Adj Close', axis=1, level=0)
df_raw_AC.to_csv(f'{data_path}/Data/Raw/top40_adj_close.csv', index=True, encoding='utf-8')

##### Saving all original multi level columns
df_raw.columns = ['_'.join(col).strip() for col in df_raw.columns.values]
df_raw.to_csv(f'{data_path}/Data/Raw/top40_complete.csv', index=True, encoding='utf-8')

#######################################################################
    # SPLAC index #
##### Getting the ticker
splac_tick =  yf.Ticker("^SPLAC")

##### Download and Collection
start_date = "2000-07-30"
    #SPLAC Launch data = September 30, 1999
end_date = "2026-08-31"
    #Top 40 recorded on 8/31/26
df_splac_raw = splac_tick.history(interval='1d', start = end_date, end=end_date)

##### SAve only 'Adj Close'
df_splac_AC = df_splac_raw.xs('Adj Close', axis=1, level=0)
df_splac_AC.to_csv(f'{data_path}/Data/Raw/splac_adj_close.csv', index=True, encoding='utf-8', multi_level_index=False)

##### Alternative, if above doesn't work
    # Go to Google Sheets and PLace the following command
    # =GOOGLEFINANCE("INDEXSP:SPLAC", "all", "2000-01-01", "2026-08-31", "DAILY")
