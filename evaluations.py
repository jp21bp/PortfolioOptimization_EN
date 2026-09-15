"""
This file contains the following from the original paper:
* Non-ML portoflio strategies
* Performance Metrics

Details:
* Risk free rate won't be incorporated for simplicity reasons
* Using simple returns, since daily returns will be used for evaluation

"""
##### Import libraries
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pypfopt import risk_models
from pypfopt import expected_returns
from scipy.optimize import minimize
import os, copy, pickle
from tqdm import tqdm
from functools import reduce

#### Reading data
### Data path
data_path = os.path.join(
    os.getcwd(),
    'Data'
)
### Prices of selected stocks
df_prices = pd.read_csv(
    f"{data_path}/Filtered/top40_adj_close.csv",
    parse_dates=['Date'],
    index_col='Date'
)
### Reading csv with all info about each stock
df_all_assets_info = pd.read_csv(
    f"{data_path}/top40_alphabetized_EN.csv"
)
### Selecting the stocks that are part of selected stocks
df_asset_infos = df_all_assets_info[
    df_all_assets_info['Ticker'].isin(df_prices.columns)
]


#### Creating simple returns
df_simple_rets = ((df_prices/df_prices.shift(1)) - 1)

#### Creating dict to capture all portfolio strats
all_port_returns = {}

#### Global Hyperparams
NUM_STOCKS = len(df_prices.columns)
STOCK_NAMES = df_prices.columns
WINDOW_SIZE = 50

################################################
    # Volatility scaling on Returns #
    # Used to focus on strategy rather than market volatility
#### Hyperparams
VOL_SCALE = 0.1
#### Calculating EWMSD
    # EWMSD = Exponentially Weighted Moving Std. Dev.
df_ewmsd_daily = df_simple_rets\
    .ewm(span=WINDOW_SIZE, adjust=False).std()  #adjust: True => EWMA vs False => EMA

df_ewmsd_annual = df_ewmsd_daily * np.sqrt(252)

#### Scaling EWMSD
df_ewmsd_scaled = (VOL_SCALE/(df_ewmsd_annual + 1e-8)).shift(1)
    # "shift(1)" to prevent look-ahead bias

#################################################
    # Performance Metrics #
##### Function
def performance_metrics(ds_port_returns: pd.Series, periodic_rate: int = 252) -> dict:
    # "ds_port_returns".shape = (num_days,)
    assert type(ds_port_returns) == pd.Series
    # Base Case
    if ds_port_returns.size == 0:
        return {
            "E(R)": 0.0,    # Annualized Return
            "Std(R)": 0.0,  # Annualized Volatility
            "Sharpe": 0.0,  # Sharpe Ratio
            "DD": 0.0,      # Downside Deviation
            "Sortino": 0.0, # Sortino Ratio
            "MD": 0.0,      # Max Drawdown
            "% Pos.(R)": 0.0,   #Percent of Positive Returns
            "P/L Ratio": 0.0,   # Profit/Loss ratio
            "Compounded Returns": np.array([1.0])   #Compounded Returns
        }                       
    
    # Annualize return
    mean_daily_ret = ds_port_returns.mean()
    annualized_ret = mean_daily_ret * periodic_rate

    # Annualized volatility
    vol_daily_ret = ds_port_returns.std()
    annualized_vol = vol_daily_ret * np.sqrt(periodic_rate)
    
    # Sharpe ratio
    annualized_sharpe = annualized_ret/(annualized_vol + 1e-8)

    # Downside deviation
    neg_rets = ds_port_returns[ds_port_returns < 0]
    annualized_downside_dev = neg_rets.std() * np.sqrt(periodic_rate)\
        if len(neg_rets) > 0 else 0.0

    # Sortino ratio
    annualized_sortino = annualized_ret/(annualized_downside_dev + 1e-8)\
        if annualized_downside_dev > 0.0 else 0.0
    
    # Compounded returns
    compounded_rets = (1 + ds_port_returns).cumprod()

    # Max Drawdown
    peak = np.maximum.accumulate(compounded_rets.values)
    drawdown = (compounded_rets - peak)/(peak + 1e-8)
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Percentage of positive returns
    per_pos_rets = (len(ds_port_returns[ds_port_returns > 0])/ ds_port_returns.shape[0]) * 100 \
        if len(ds_port_returns) > 0 else 0.0

    # P/L Ratio
    pos_rets = ds_port_returns[ds_port_returns > 0]
    neg_rets = ds_port_returns[ds_port_returns < 0]
    avg_profit = pos_rets.mean() if len(pos_rets) > 0 else 0.0
    avg_loss = neg_rets.mean() if len(neg_rets) > 0 else 0.0
    pl_ratio = abs(avg_profit/(avg_loss + 1e-8)) if avg_loss < 0.0 else 0.0

    # Results
    return {
        "E(R)": annualized_ret,
        "Std(R)": annualized_vol,
        "Sharpe": annualized_sharpe,
        "DD": annualized_downside_dev,
        "Sortino": annualized_sortino,
        "MD": max_drawdown,
        "% Pos.(R)": per_pos_rets,
        "P/L Ratio": pl_ratio,
        "Compounded Returns": compounded_rets
    }


#################################################
    # Fixed allocation strategy #
    # Allocations will be based on sector #
#### Strategy function
def fixed_alloc(df_asset_returns: pd.DataFrame, vol_scaling: bool, fixed_weights: np.ndarray, sector: str) -> pd.Series:
    # "df_asset_returns".shape = (num_days, num_assets)
    # "fixed_weights".shape = (num_assets,)
        # These weights are fixed for the entire portfolio
    # return: np_port_rets, with shape = (num_days)

    # Error check
    assert df_asset_returns.shape[1] == fixed_weights.shape[0]

    # Integrating vol_scaling
    if vol_scaling: df_scaled_weights = df_ewmsd_scaled * fixed_weights
    else: df_scaled_weights = fixed_weights

    # Calculating portfolio returns
    np_port_rets = (df_asset_returns * df_scaled_weights).sum(axis=1)
        # Shape = (num_days,)

    # Error check
    assert type(np_port_rets) == pd.Series
    np_port_rets.name = f'FA_{sector}_returns'
        
    return np_port_rets

#### Setup: Stock and Sector relationship
### Selecting top 4 sectors amoung stocks
top4_sectors = df_asset_infos['Sector']\
    .value_counts(sort=True, ascending=False)\
    .index.tolist()[:4]
### Mapping sector to the number of stocks in that sector
dict_sector_stock_count = df_asset_infos['Sector']\
    .value_counts().to_dict()
### Mapping the stock to its sector
dict_stock_to_sector = dict(zip(
    df_asset_infos['Ticker'],
    df_asset_infos['Sector']
))

#### Implementation
    # Create a separate allocations for each of the top 4
    # In each allocation, the chosen sector = 60%, rest = 40%
alloc_strat_returns = []
for sector in top4_sectors:
    # Dividing stocks
    num_sect_stock = dict_sector_stock_count[sector]
    num_nonsect_stock = NUM_STOCKS - num_sect_stock
    # Weight alloc
    sect_weight_alloc = 0.6/num_sect_stock
    nonsect_weight_alloc = 0.4/num_nonsect_stock
    # Creating weight
    weights = []
    for stock in STOCK_NAMES:
        sect = dict_stock_to_sector[stock]
        weights.append(
            sect_weight_alloc if sect == sector else nonsect_weight_alloc
        )
    weights = np.array(weights)
    assert np.sum(weights).round(2) == 1.00
    # Implement strategy
    ds_port_ret_noVS = fixed_alloc(df_simple_rets, False, weights, sector).dropna()
    ds_port_ret_VS = fixed_alloc(df_simple_rets, True, weights, sector).dropna()
    all_port_returns[f'FA_{sector}_noVS'] = ds_port_ret_noVS
    all_port_returns[f'FA_{sector}_VS'] = ds_port_ret_VS

######################################################
    # Mean-Variance Optimization #
#### Function definition
def MVO(df_asset_returns: pd.DataFrame, vol_scaling: bool) -> pd.Series:
    # "df_asset_returns".shape = (num_days, num_assets)
    # Creating dataframe
    ds_port_returns = pd.Series(name='MVO_returns', index = df_asset_returns.index)

    # Iterating through all windows
    for t in tqdm(range(WINDOW_SIZE, df_asset_returns.shape[0])):
        # Setup
        window_rets = df_asset_returns.iloc[t-WINDOW_SIZE:t]
        win_mean_rets = window_rets.mean(axis=0)
        win_cov_rets = window_rets.cov()

        # MVO
        ef = EfficientFrontier(win_mean_rets, win_cov_rets)
        try:
            weights = ef.max_sharpe(risk_free_rate=0.0)
            clean_weights = ef.clean_weights()
                # "clean_weights".shape ~ (num_assets,)
            np_clean_weights = np.array(list(clean_weights.values()))
        except:
            np_clean_weights = np.ones(df_asset_returns.shape[1])/ df_asset_returns.shape[1]

        # Integrting volatility scaling
        if vol_scaling: scaled_weights = df_ewmsd_scaled.iloc[t] * np_clean_weights
        else: scaled_weights=np_clean_weights

        # Portfolio return
        port_ret = np.dot(df_asset_returns.iloc[t].values, scaled_weights)
        ds_port_returns.loc[df_asset_returns.iloc[t].name] = port_ret

    return ds_port_returns

#### Implementation
pickle_path = os.path.join(
    os.getcwd(),
    '2_OptimizacionPortafolio',
    'Pickles'
)
### MVO no Vol. Scaling
path = f'{pickle_path}/MVO_port_ret_noVS.pkl'
if os.path.isfile(path):
    with open(path, 'rb') as file:
        ds_mvo_port_ret_noVS = pickle.load(file)
else:
    ds_mvo_port_ret_noVS = MVO(df_simple_rets, False)
    pickle.dump(ds_mvo_port_ret_noVS, open(path, 'wb'))

### MVO with Vol. Scaling
path = f'{pickle_path}/MVO_port_ret_VS.pkl'
if os.path.isfile(path):
    with open(path, 'rb') as file:
        ds_mvo_port_ret_VS = pickle.load(file)
else:
    ds_mvo_port_ret_VS = MVO(df_simple_rets, False)
    pickle.dump(ds_mvo_port_ret_VS, open(path, 'wb'))
### Collection
all_port_returns['MVO_noVS'] = ds_mvo_port_ret_noVS
all_port_returns['MVO_VS'] = ds_mvo_port_ret_VS


######################################################
    # Maximum Diversification Optimization #
#### Function definition
def MDO(df_asset_returns: pd.DataFrame, vol_scaling: bool) -> pd.Series:
    # "np_asset_returns".shape = (num_days, num_assets)

    # Creating data Series
    ds_port_returns = pd.Series(name='MDO_returns', index = df_asset_returns.index)

    # Initializing weights
    num_assets = df_asset_returns.shape[1]
    last_weights = np.ones(num_assets)/num_assets

    for t in tqdm(range(WINDOW_SIZE, df_asset_returns.shape[0])):
        # Setup
        window_rets = df_asset_returns.iloc[t-WINDOW_SIZE:t]
        win_cov_rets = window_rets.cov()
        win_asset_vol = window_rets.std(axis=0)
        win_asset_vol = np.maximum(win_asset_vol, 1e-6)
            #Ensure win_asset_vol > 0

        # Objective function
        def objective(weights : np.ndarray):
            port_vol = np.sqrt(np.dot(weights.T, np.matmul(win_cov_rets, weights)))
            weighted_asset_vol = np.dot(win_asset_vol, weights)
            diversification_ratio = weighted_asset_vol / (port_vol + 1e-8)
            return -diversification_ratio
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) # Sum of weights = 1
        bounds = tuple((0, 1) for _ in range(num_assets)) # Weights between 0 and 1

        # Solve optimization problem
        optimal_weights = minimize(
            objective,
            last_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # Ensure valid weights
        if (not optimal_weights.success) or \
            (np.any(np.isnan(optimal_weights.x))):
            weights = np.ones(num_assets)/ num_assets
        else:
            weights = optimal_weights.x
            weights = weights/np.sum(weights)   #Normalize
        last_weights = weights
        if vol_scaling: scaled_weights = df_ewmsd_scaled.iloc[t] * weights
        else: scaled_weights = weights

        # Portfolio return
        port_ret = np.dot(df_asset_returns.iloc[t], scaled_weights)
        ds_port_returns.loc[df_asset_returns.iloc[t].name] = port_ret

    return ds_port_returns

#### Implementation
pickle_path = os.path.join(
    os.getcwd(),
    '2_OptimizacionPortafolio',
    'Pickles'
)
### MDO no Vol. Scaling
path = f'{pickle_path}/MDO_port_ret_noVS.pkl'
if os.path.isfile(path):
    with open(path, 'rb') as file:
        ds_mdo_port_ret_noVS = pickle.load(file)
else:
    ds_mdo_port_ret_noVS = MVO(df_simple_rets, False)
    pickle.dump(ds_mdo_port_ret_noVS, open(path, 'wb'))

### MVO with Vol. Scaling
path = f'{pickle_path}/MDO_port_ret_VS.pkl'
if os.path.isfile(path):
    with open(path, 'rb') as file:
        ds_mdo_port_ret_VS = pickle.load(file)
else:
    ds_mdo_port_ret_VS = MVO(df_simple_rets, False)
    pickle.dump(ds_mdo_port_ret_VS, open(path, 'wb'))
### Collection
all_port_returns['MDO_noVS'] = ds_mdo_port_ret_noVS
all_port_returns['MDO_VS'] = ds_mdo_port_ret_VS

###############################################
    # ML strats #
#### Setting models' path
developed_path = os.path.join(
    os.getcwd(),
    '2_OptimizacionPortafolio',
    'DevelopedModels'
)

#### Function for processing
def ml_rets(df_weights: pd.DataFrame, vol_scaling: bool):
    # Volatility scaling
    if vol_scaling: 
        df_ewmsd_scaled_model = df_ewmsd_scaled.loc[df_weights.index]
        df_scaled_weights = df_ewmsd_scaled_model * df_weights.values
    else:
        df_scaled_weights = df_weights

    # Selecting simple returns
    df_simple_rets_models = df_simple_rets.loc[df_scaled_weights.index]

    # Calculating portfolio returns
    ds_port_rets = (df_simple_rets_models * df_scaled_weights.values).sum(axis=1)

    return ds_port_rets


#### Iterating through all models
names = ['Slide_P_LR', 'Slide_T_H_O', 'Slide_All', 
         'Expand_P_LR', 'Expand_T_H_O', 'Expand_All']
for i in range(1,7):
    # Extracting weight predictions from current model
    path = f'{developed_path}/Model{i}'
    df_weights_results = pd.read_csv(
        f'{path}/weight_results.csv',
        parse_dates=[0],
        index_col=0
    )
    df_weights_only = df_weights_results[df_weights_results.columns[:-1]]

    # Getting returns
    model_rets_noVS = ml_rets(df_weights_only, False)
    all_port_returns[f'Model_{names[i-1]}_noVS'] = model_rets_noVS

    model_rets_VS = ml_rets(df_weights_only, True)
    all_port_returns[f'Model_{names[i-1]}_VS'] = model_rets_VS


###############################################
    # Benchmark #
### Reading data
df_bench = pd.read_csv(
    f'{data_path}/raw_splac_prices.csv',
    parse_dates=['Date'],
    index_col='Date'
)
ds_bench_prices = df_bench['Close']
ds_bench_returns = ((ds_bench_prices/ds_bench_prices.shift(1)) - 1).dropna()
### Transformando el index del benchmark
ds_bench_returns.index = ds_bench_returns.index.normalize()

#### Joining data
all_port_returns['Benchmark_noVS'] = ds_bench_returns
all_port_returns['Benchmark_VS'] = ds_bench_returns
all_port_returns['Model_Benchmark_noVS'] = ds_bench_returns
all_port_returns['Model_Benchmark_VS'] = ds_bench_returns

#### Identifying comons dates
common_dates = reduce(
    lambda x,y: x.intersection(y), 
    [ds.index for ds in all_port_returns.values()]
)
str_common_dates = [f'{date.day:02d}-{date.month:02d}\n-{date.year}' for date in common_dates]

###############################################
    # Evaluation #
#### Graphic visual 
### Setup
all_port_results = {}
per_results_ML_VS = []
per_results_ML_noVS = []
per_results_noML_VS = []
per_results_noML_noVS = []
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16,12))
colors = ['#1f77b4','#1f77b4', "#ff7f0e", "#ff7f0e", "#2ca02c", "#2ca02c", 
          "#d62728","#d62728",  "#9467bd","#9467bd", "#8c564b", "#8c564b",
          "#e377c2","#e377c2",  "#7f7f7f", "#7f7f7f", "#bcbd22","#bcbd22",
          "#17becf", "#17becf", "#ffbb78","#ffbb78",  "#98df8a", "#98df8a"]
### Graph
for i, (strat, ds_port_ret) in enumerate(all_port_returns.items()):
    # Calculating return performance
    performance = performance_metrics(ds_port_ret.loc[common_dates])
    all_port_results[strat] = copy.deepcopy(performance)
    strat_split = strat.split("_")
    performance['Strategy'] = "_".join([word for word in strat_split[:-1]])
    if strat_split[0] == 'Model': performance['Strategy'] = "_".join([word for word in strat_split[1:-1]])
    if strat_split[0].startswith('Model'):  # ML strats
        if strat_split[-1] == 'noVS':
            axs[0,0].plot(
                range(len(common_dates)),
                performance['Compounded Returns'],
                color = colors[i] if strat_split[-2] != 'Benchmark' else 'black',
                label = performance['Strategy']
            )
            # Appropriately store performance
            del performance['Compounded Returns']
            per_results_ML_noVS.append(performance)
        else:
            axs[0,1].plot(
                range(len(common_dates)),
                performance['Compounded Returns'],
                color = colors[i] if strat_split[-2] != 'Benchmark' else 'black',
                label = performance['Strategy']
            )
            # Appropriately store performance
            del performance['Compounded Returns']
            per_results_ML_VS.append(performance)
    else:   # nonML strats
        if strat_split[-1] == 'noVS':
            axs[1,0].plot(
                range(len(common_dates)),
                performance['Compounded Returns'],
                color = colors[i] if strat_split[-2] != 'Benchmark' else 'black',
                label = performance['Strategy']
            )
            # Appropriately store performance
            del performance['Compounded Returns']
            per_results_noML_noVS.append(performance)
        else:
            axs[1,1].plot(
                range(len(common_dates)),
                performance['Compounded Returns'],
                color = colors[i] if strat_split[-2] != 'Benchmark' else 'black',
                label = performance['Strategy']
            )
            # Appropriately store performance
            del performance['Compounded Returns']
            per_results_noML_VS.append(performance)
# Final configs
for ax in axs.flatten():
    ax.set_xticks(range(len(common_dates)))
    ax.set_xticklabels(str_common_dates, rotation=90)
    ax.xaxis.set_major_locator(MultipleLocator(400))
    ax.legend(fontsize = 6.5)
axs[0,0].set_title('ML Models without Vol. Scaling', fontweight='bold')
axs[0,1].set_title('ML Models with Vol. Scaling', fontweight='bold')
axs[1,0].set_title('Non-ML Models without Vol. Scaling', fontweight='bold')
axs[1,1].set_title('Non-ML Models with Vol. Scaling', fontweight='bold')
fig.supxlabel('Dates', x = 0.5, y=0, fontweight='bold')
fig.supylabel('Compounded Return', x = 0.075, y = 0.5, fontweight='bold')
plt.subplots_adjust(hspace=0.4)
plt.show()



#### Numeric visual
## Creating dataframes
metrics = ['Strategy', 'E(R)', 'Std(R)', 'Sharpe',
           'DD', 'Sortino', 'MD',
           '% Pos.(R)', 'P/L Ratio']
df_results_ML_noVS = pd.DataFrame(per_results_ML_noVS, columns=metrics)
df_results_ML_VS = pd.DataFrame(per_results_ML_VS, columns=metrics)
df_results_noML_noVS = pd.DataFrame(per_results_noML_noVS, columns=metrics)
df_results_noML_VS = pd.DataFrame(per_results_noML_VS, columns=metrics)

# df_results.sort_values(by='Ratio Sharpe', ascending=False)

### Creating numeric visual
## Setup
fig, axs = plt.subplots(ncols=2, nrows=2, figsize = (8,8))
dfs = [df_results_ML_noVS, df_results_ML_VS, df_results_noML_noVS, df_results_noML_VS]
titles = ['ML Models without Vol. Scaling', 'ML Models with Vol. Scaling',
          'Non-ML Models without Vol. Scaling', 'Non-ML Models with Vol. Scaling']

## Creating tables
for ax, df_data, title in zip (axs.flat, dfs, titles):
    ax.axis('off')
    table = ax.table(
        cellText=df_data.round(3).values,
        colLabels=df_data.columns,
        rowLabels = None,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    # Ajusting the col widths
    for row in range(len(df_data) + 1):  # +1 para incluir encabezado
        table[(row, 0)].set_width(0.35)  
        table[(row, 1)].set_width(0.1) 
        table[(row, 2)].set_width(0.1) 
        table[(row, 3)].set_width(0.12) 
        table[(row, 4)].set_width(0.1) 
        table[(row, 5)].set_width(0.12) 
        table[(row, 6)].set_width(0.1) 
        table[(row, 7)].set_width(0.15) 
        table[(row, 8)].set_width(0.15) 
    # Finding optimal values
    max_vals = []
    for col in df_data.columns:
        if col == 'Strategy': continue
        if (col == 'Std(R)') or (col == 'DD'):
            max_vals.append(df_data[col].min())
        else: max_vals.append(df_data[col].max())
    # Highlighting
    for j in range(1, df_data.shape[1]):
        for i in range(df_data.shape[0]):
            cell = table[i+1, j]
            if df_data.iat[i, j] == max_vals[j-1]:
                cell.set_facecolor('yellow')
    # Adding title
    ax.text(0.5, 0.8, title, ha="center", va="bottom",
        fontsize=14, fontweight="bold", transform=ax.transAxes)

plt.tight_layout()
plt.show()
