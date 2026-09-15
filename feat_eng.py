"""
Data feature engineering

Focus technical indicators :
* Prices
* Log returns
* TEMA (Triple Exponential Moving Average)
* HLC3 (High Low Close 3-average)
* OBV (On Balance Volume)

Model details:
* Three separate models with the same input structure:
    - Input shape = (t, k * n)
        * t = lookback window = 50 time steps
        * n = number of assets = 21 stocks
        * k = number of tecnical indicators
* Model 1 tecnical indicators (similar to paper)
    - Prices
    - Log returns
* Model 2 tecnical indicators:
    - HLC3
    - OBV
    - TEMA
* Model 3 tecnical indicators:
    - All 5 of the above
* Thus, there will be three different datasets pre-processed
"""
#### Importacion de modulos
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy
import os

#### Glocal hyperparams
WINDOW_SIZE = 50

#### Data path
data_path = os.path.join(
    os.getcwd(),
    'Data',
    'Filtered'
)

#### Reading top 40 all info
df_all = pd.read_csv(
    f"{data_path}/top40_complete.csv",
    parse_dates=['Date'],
    index_col='Date'
)

#### Tecnical indicator 1 : Prices
df_prices = pd.read_csv(
    f"{data_path}/top40_adj_close.csv",
    parse_dates=['Date'],
    index_col='Date'
)

#### Tecnical indicator 2: log returns
df_log_rets = np.log(df_prices/df_prices.shift(1))

######################################################
    # Tecnical indicator 3: TEMA #
    # TEMA = 3*EMA(p) - 3*EMA(EMA(p)) + EMA(EMA(EMA(p)))
        # EMA = Exponential Moving Average
#### Calculating EMAs
df_ema_1 = df_prices\
    .ewm(span=WINDOW_SIZE, adjust=False)\
    .mean()

df_ema_2 = df_ema_1\
    .ewm(span=WINDOW_SIZE, adjust=False)\
    .mean()

df_ema_3 = df_ema_2\
    .ewm(span=WINDOW_SIZE, adjust=False)\
    .mean()

#### TEMA
df_tema = 3*df_ema_1 - 3*df_ema_2 + df_ema_3


######################################################
    # HLC3 #
#### Creating DF 
df_hlc3 = pd.DataFrame(
    columns=df_prices.columns,
    index=df_prices.index
)

#### Calculating HLC3 for each stock
for stock in df_prices.columns:
    df_stock_hlc3 = df_all[[f'High_{stock}', f'Low_{stock}', f'Close_{stock}']]
    assert df_stock_hlc3.shape[1] == 3
    df_hlc3[f'{stock}'] = (df_stock_hlc3.sum(axis=1))/3

###########################################################
    # OBV #
#### Creating OBV
df_obv = pd.DataFrame(
    columns=df_prices.columns,
    index=df_prices.index
)

#### Calculating OBV for each stock
for stock in df_prices.columns:
    # Seleccing necessary info 
    df_stock_obv = df_all[[f'Volume_{stock}', f'Close_{stock}']]
    assert df_stock_obv.shape[1] == 2
    # Caculating OBV
    obv = [0]
    for i in range(1,len(df_stock_obv)):
        if df_stock_obv[f'Close_{stock}'].iloc[i] > df_stock_obv[f'Close_{stock}'].iloc[i - 1]:
            obv.append(obv[-1] + df_stock_obv[f'Volume_{stock}'].iloc[i])
        elif df_stock_obv[f'Close_{stock}'].iloc[i] < df_stock_obv[f'Close_{stock}'].iloc[i - 1]:
            obv.append(obv[-1] - df_stock_obv[f'Volume_{stock}'].iloc[i])
        else:
            obv.append(obv[-1])  
    df_obv[f'{stock}'] = obv

###########################################################
    # Aggregating all Data #
#### Renaming each DF
df_prices.columns = [f'Price_{stock}' for stock in df_prices.columns]
df_log_rets.columns = [f'LogRet_{stock}' for stock in df_log_rets.columns]
df_tema.columns = [f'TEMA_{stock}' for stock in df_tema.columns]
df_hlc3.columns = [f'HLC3_{stock}' for stock in df_hlc3.columns]
df_obv.columns = [f'OBV_{stock}' for stock in df_obv.columns]

#### Joining DFs
df_technical_indicators = pd.concat(
    [df_prices, df_log_rets, df_tema, df_hlc3, df_obv],
    axis=1
)

#### Deleting NaNs
df_technical_indicators = df_technical_indicators.dropna(axis=0)

#### SAving tecnical indicators
df_technical_indicators.to_csv(
    f"{data_path}/technical_indicators.csv",
    index=True,
    encoding='utf-8'
)









