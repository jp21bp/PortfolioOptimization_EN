"""
This file will clean the following two raw datas:
* raw_top40.csv: the "Adj Close" of the top 40 companies in SPLAC
* raw_top40_complete.csv: all components of the top 40 companies in SPLAC
"""

##### Importing libraries
import pandas as pd
import os 

##### Data path
data_path = os.path.join(
    os.getcwd(),
    'Data',
    'Raw'
)

##### Reading datas
df = pd.read_csv(
    f'{data_path}/top40_adj_close.csv',
    parse_dates=['Date'],
    index_col='Date'
)

df_complete = pd.read_csv(
    f'{data_path}/top40_complete.csv',
    parse_dates=['Date'],
    index_col='Date'
)


#######################################
    # raw_top40.csv - Nulls #
#### Forward fill (for weekends) before full null-analysis
df = df.ffill()

#### Capturing 2008 crisis and build up
    # Erasing all companies with NaN after 2006-3-31
df_erase = df[df.index > "2006-3-31"].isna().sum()
cols_to_erase = df_erase[df_erase > 0].index.to_list()
df = df.drop(columns=cols_to_erase)

#### Creating dataframe where ALL rows have some value
df = df[df.notna().all(axis=1)]

##########################################
    # raw_top40.csv - Duplicates #
#### Counting number of duplicates
df[df.duplicated()].shape
    # There are 47 repeated rows

#### Delete duplicates 
df = df.drop_duplicates()


##########################################
    # raw_top40.csv - Saving #
df.to_csv(
    f'{data_path}/Clean/top40_adj_close.csv',
    index=True,
    encoding='utf-8')

#########################################
    # raw_top40_complete.csv #    
#### Forward fill
df_complete = df_complete.ffill()

#### Working with the same dates as above
df_complete = df_complete[df_complete.index.isin(df.index)]

#### Dropping cols/stocks that aren't same as above
### Gather stockes that passed the filter
filtered_stocks = df.columns
cols_to_delete = []
### Looping to find cols that need to be deleted
for col in df_complete.columns:
    stock = col.split('_')[1]
    if stock not in filtered_stocks: 
        cols_to_delete.append(col)
### Dropping cols/stocks 
df_complete = df_complete.drop(columns=cols_to_delete)

#### Saving
df_complete.to_csv(
    f'{data_path}/Clean/top40_complete.csv',
    index=True,
    encoding='utf-8')





