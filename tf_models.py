"""
This file will create the data preprocessing and TF models

Recall: the ratio sharpe is being maximized
    Where the ratio sharpe is seen as part of the model loss function

Reason for maximizing ratio sharpe and not returns:
    Ratio sharpe takes into consideration the volatility
"""
#### Importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, pickle, copy, keyboard
from tqdm import tqdm

#### Data path
data_path = os.path.join(
    os.getcwd(),
    'Data'
)

#### Reading data
df_tech_indicators = pd.read_csv(
    f'{data_path}/technical_indicators.csv',
    parse_dates=['Date'],
    index_col='Date'
)

#### Creating simple return
    # NOT one of the indicators
y_all_prices = df_tech_indicators.filter(regex="^Price")
y_all_simple_rets = y_all_prices.pct_change()


#### Global hyperparams
WINDOW_SIZE = 50
NUM_STOCKS = 21
STOCK_NAMES = [stock for indicator, stock in \
               df_tech_indicators.columns[:21].str.split("_")]
#### Seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#####################################
    # Pre-processing #
#### General window split
def window_split(
        df_test_train_set: pd.DataFrame,
        type: str
) -> np.ndarray:
    # Set up
    all_inputs = []
    all_labels = []
    num_indicators = int(df_test_train_set.shape[1]/NUM_STOCKS)
        # Each indicator has NUM_STOCKS associated stocks

    # Splits
    for t in tqdm(
        range(WINDOW_SIZE, df_test_train_set.shape[0]),
        desc=f'IndividualSplits - {type}',
        position=1,
        leave=False
    ):
        # Extracting Data input
        X = df_test_train_set\
            .iloc[t-WINDOW_SIZE: t]
        # Normalizing data input across EACH technical indicator
        X_norm = pd.DataFrame(
            columns=X.columns,
            index=X.index
        )
        for i in range(num_indicators):
            # Selecting corresponding stocks
            cols = X.columns[i*NUM_STOCKS: (i+1)*NUM_STOCKS]
            indicator_subset = X[cols]
            subset_vals = indicator_subset.values
            # Normalizing indicator
            X_norm[cols] = (subset_vals - np.mean(subset_vals)) / \
                (np.std(subset_vals) + np.finfo(float).eps)
        # Extracting label = future stock simple return of a given day
        y = y_all_simple_rets.iloc[t] 
        # Concatenation
        all_inputs.append(X_norm.values)
        all_labels.append(y.values)

    # Recording dates
        # Useful when graphing model results
    split_label_dates = df_test_train_set.iloc[WINDOW_SIZE: df_test_train_set.shape[0]].index

    return [np.array(all_inputs), np.array(all_labels), split_label_dates.values]

#### Sliding window function
def sliding_window_split(
        df_features: pd.DataFrame,
        train_years: float = 1.0,
        val_years: float = 0.25,
        test_years: float = 1.0,
) -> np.ndarray:
    # Set up
    days_per_year = 252
    train_days = int(train_years * days_per_year)
    val_days = int(val_years * days_per_year)
    test_days = int(test_years * days_per_year)
    train_test_sets = {}
    fullset_counter = 0
    last_day_for_full_set = df_features.shape[0] \
        - test_days - val_days - train_days
    for train_start in tqdm(
        range(0, last_day_for_full_set, test_days - WINDOW_SIZE),
        desc='SlidingTrainTestSplit',
        position=0
    ):
        # Data start dates
        val_start = train_start + train_days
        test_start = val_start + val_days

        # Edge case
        if (test_start + test_days) >  df_features.shape[0]:
            break   # last set runs past avaialble days

        # Extracting data
        df_train_data = df_features\
            .iloc[train_start: train_start + train_days]
        df_val_data = df_features\
            .iloc[val_start: val_start + val_days]
        df_test_data = df_features\
            .iloc[test_start: test_start + test_days]

        # Splitting each set
        train_data_splits = window_split(df_train_data, "train")
        val_data_splits = window_split(df_val_data, "val")
        test_data_splits = window_split(df_test_data, "test")

        # Appending data
        train_test_sets[f'fullset{fullset_counter}_train_val_test_list'] = [train_data_splits, val_data_splits, test_data_splits]
        fullset_counter += 1

    return train_test_sets

#### Expanding window split
def expanding_window_split(
        df_features: pd.DataFrame,
        train_years: float = 1.0,
        val_years: float = 0.25,
        test_years: float = 1.0,
) -> np.ndarray:
    # Set up
    days_per_year = 252
    train_days = int(train_years * days_per_year)
    val_days = int(val_years * days_per_year)
    test_days = int(test_years * days_per_year)
    train_test_sets = {}
    fullset_counter = 0
    last_day_for_full_set = df_features.shape[0] \
        - val_days - test_days 
    for train_end in tqdm(
        range(train_days, last_day_for_full_set, test_days - WINDOW_SIZE),
        desc='ExpandingTrainTestSplit',
        position=0
    ):    
        # Data start dates
        val_start = train_end
        test_start = val_start + val_days

        # Edge case
        if (test_start + test_days) >  df_features.shape[0]:
            break   # last set runs past avaialble days

        # Extracting data
        df_train_data = df_features\
            .iloc[:train_end]
        df_val_data = df_features\
            .iloc[val_start: val_start + val_days]
        df_test_data = df_features\
            .iloc[test_start: test_start + test_days]

        # Splitting each set
        train_data_splits = window_split(df_train_data, "train")
        val_data_splits = window_split(df_val_data, "val")
        test_data_splits = window_split(df_test_data, "test")

        # Appending data
        train_test_sets[f'fullset{fullset_counter}_train_val_test_list'] = [train_data_splits, val_data_splits, test_data_splits]
        fullset_counter += 1

    return train_test_sets



#### Technical indicator selection
    # Choosing specific indicators instead of all 5
    # Each indicator is NUM_STOCKS cols/stocks long
    # Selection will happen post splits from above
def indicator_selection(list_indicators: list, split_strat: dict):
    ### Making deep copy
    split_strat_cpy = copy.deepcopy(split_strat)

    ### Mapping: indicator -> corresponding cols
        # Based on their order in the original DF
    ranges = []
    for indicator in list_indicators:
        if indicator == 'Price':
            ranges.append((0,21))
        elif indicator == 'LogRet':
            ranges.append((21,42))
        elif indicator == 'TEMA':
            ranges.append((42,63))
        elif indicator == 'HLC3':
            ranges.append((63,84))
        elif indicator == 'OBV':
            ranges.append((84,105))
        else:
            raise TypeError('Indicator not written correctly')
        
    ### Creating col indices
    cols = np.concatenate(
        [np.arange(start, end) for start, end in ranges]
    )

    ### Selecting indicators
    for fullset_key, train_val_test_list in tqdm(
        split_strat_cpy.items(),
        total=len(split_strat_cpy),
        desc='FullsetDict',
        position=0
    ):
        for tvt_set in train_val_test_list:
            # Each tvt_set = [np_inputs, np_labels, np_dates]
            tvt_set[0] = tvt_set[0][:,:,cols]
                # Recall: no_inputs.shape = (202 windows, 50 days, 105 cols/stock_indicators)

    return split_strat_cpy


################################################
    # Implementing data pre-processing #
#### Implementation
### Sliding strat
sliding_strat_path = f'{data_path}/Pickles/dict_sliding_strat.pkl'
if os.path.isfile(sliding_strat_path):
    with open(sliding_strat_path, 'rb') as file:
        dict_sliding_fullsets = pickle.load(file)
else:
    dict_sliding_fullsets = sliding_window_split(df_tech_indicators)
    pickle.dump(dict_sliding_fullsets, open(sliding_strat_path, 'wb'))

### Expanding strat
expanding_strat_path = f'{data_path}/Pickles/dict_expanding_strat.pkl'
if os.path.isfile(expanding_strat_path):
    with open(expanding_strat_path, 'rb') as file:
        dict_expanding_fullsets = pickle.load(file)
else:
    dict_expanding_fullsets = expanding_window_split(df_tech_indicators)
    pickle.dump(dict_expanding_fullsets, open(expanding_strat_path, 'wb'))


#####################################################
    # TF Model#
#### Creating initializers y regularizer
glorot_init = tf.keras.initializers.GlorotUniform(seed=SEED)
orthogonal_init = tf.keras.initializers.Orthogonal(seed=SEED)
zero_init = tf.keras.initializers.Zeros()
regularizer = tf.keras.regularizers.l2(1e-5)
DROPOUT = 0.2
LEARN_RATE = 0.001
#### Creating Model class
class LSTMModel(tf.keras.Model):
    def __init__(
        self, 
        num_indicators, 
        num_assets, 
        **kwargs
    ):
        super(LSTMModel, self).__init__(**kwargs)
        self.num_indicators = num_indicators
        self.num_assets=num_assets
        self.num_filters_units= 64
        self.cnn = tf.keras.layers.Conv1D(
            filters = self.num_filters_units,
            kernel_size=5,
            padding="same",
            data_format='channels_last',
            activation='relu',
            kernel_initializer = glorot_init,
            bias_initializer = zero_init,
        )
        self.lstm1 = tf.keras.layers.LSTM(
            units = self.num_filters_units,
            input_shape = (WINDOW_SIZE, num_indicators * num_assets),
            kernel_initializer = glorot_init,
            recurrent_initializer = orthogonal_init,
            bias_initializer = zero_init,
            return_sequences=True,
            dropout=DROPOUT,
            seed=SEED,
            kernel_regularizer = regularizer,
            name="lstm_1"
        )
        self.avg = tf.keras.layers.Average()
        self.lstm2 = tf.keras.layers.LSTM(
            32,
            kernel_initializer = glorot_init,
            recurrent_initializer = orthogonal_init,
            bias_initializer = zero_init,
            return_sequences=False,
            dropout=DROPOUT,
            seed=SEED,
            kernel_regularizer = regularizer,
            name="lstm_2"
        )
        self.dense = tf.keras.layers.Dense(
            num_assets,
            activation='softmax',
            kernel_initializer = glorot_init,
            bias_initializer = zero_init,
            kernel_regularizer = regularizer,
            name='dense'
        )
        self.dropout = tf.keras.layers.Dropout(
            DROPOUT,
            seed=SEED,
            name='dropout'
        )

    def call(self, inputs, training=False):
        # Add training to all layers with dropouts
        x = self.lstm1(inputs, training=training)
        x = self.dropout(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return x

    def build(self):
        dummy_input = tf.zeros((1, WINDOW_SIZE, self.num_indicators * self.num_assets))
        self.call(dummy_input)
        return
    

#### Creating weight resetter
def seed_reset_weights(model):
    tf.keras.backend.clear_session(free_memory=True)
    for layer in model.layers:
        for weight in layer.weights:
            if weight.name == 'kernel':
                weight.assign(glorot_init(shape=weight.shape))
            elif weight.name == 'recurrent_kernel':
                weight.assign(orthogonal_init(shape=weight.shape))
            elif weight.name == 'bias':
                weight.assign(zero_init(shape=weight.shape))
            else:
                print('OTHER WEIGHT TYPE')
    return model

##########################################################
    # TF Loss Function #
#### Creating loss class
class MinRS(tf.keras.losses.Loss):
    def __init__(self, name = None, reduction = "mean", dtype=None, **kwargs):
        # Reductions: {None, 'mean_with_sample_weight', 'none', 'sum_over_batch_size', 'mean', 'sum'}
        super(MinRS, self).__init__(name, reduction, dtype, **kwargs)

    def call(self, y_true, y_pred):
        # Convert to TF objects
        tf_y_true =tf.convert_to_tensor(y_true, dtype=tf.float32)   # Stock prices
        tf_y_pred =tf.convert_to_tensor(y_pred, dtype=tf.float32)   # Predicted weights
        tf_batch_all_stock_returns = tf.multiply(tf_y_true, tf_y_pred)

        ## Calculating daily ratio sharpe
            # Reason for daily: labels are daily
        # Daily returns
        days_per_year = float(252)
        batch_port_daily_returns = tf.reduce_sum(tf_batch_all_stock_returns, axis=1)
        
        # Expected returns
        daily_returns_mean = tf.reduce_mean(batch_port_daily_returns)
        annualized_returns_mean = daily_returns_mean * days_per_year
        
        # Volatility
        daily_returns_std = tf.math.reduce_std(batch_port_daily_returns)
        annualized_returns_std = daily_returns_std * tf.math.sqrt(days_per_year)

        batch_diario_rs = daily_returns_mean/(daily_returns_std + tf.keras.backend.epsilon())
        batch_annualized_rs = annualized_returns_mean/(annualized_returns_std + tf.keras.backend.epsilon())
            # To much bias for the small amount of data in the batch size

        return -batch_diario_rs

#########################################################
    # Callback #
#### Creating Callback class
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_details_path: str, lr_patience = 3, stop_patience = 10, thresh = 1e-5):
        super(CustomCallback, self).__init__()
        # Detail log path
        self.train_details_path = train_details_path
        # Threshold
        self.thresh = thresh
        # Changing learning rate
        self.lr_wait = 0
        self.lr_patience = lr_patience
        # Early stopping
        self.stop_patience = stop_patience
        self.stop_wait = 0
        # Values from previous epoch
        self.prev_val_rs = None
        self.prev_train_rs = None
        self.sum_val_rs = 0
        self.sum_train_rs = 0
        self.sum_diff_val_rs = 0
        self.sum_diff_train_rs = 0

    def on_epoch_begin(self, epoch: int, logs = None):
        return

    def on_batch_end(self, batch, logs = None):
        return

    def on_train_batch_end(self, batch: int, logs = None):
        # Hotkey stop
        if global_stop_training:
            print('STOPPING AT ', batch)
            self.model.stop_training = True
        return 
    
    def on_test_batch_end(self, batch: int, logs = None):
        return

    def on_epoch_end(self, epoch: int, logs = None):
        ##### Working with Ratio Sharpe (RS) metrics
        logs = logs or {}
        val_rs = logs.get('val_RS')
        train_rs = logs.get('RS')

        # Sum for running average
        self.sum_val_rs += val_rs
        self.sum_train_rs += train_rs

        # Registering first epoch values
        if not self.prev_val_rs:
            self.prev_val_rs = val_rs
            self.prev_train_rs = train_rs
            return

        # Calculating differences
        diff_val_rs = val_rs - self.prev_val_rs
        diff_train_rs = train_rs - self.prev_train_rs
        self.sum_diff_val_rs += diff_val_rs
        self.sum_diff_train_rs += diff_train_rs

        # Case: validation ratio sharpe isn't improving -> case for early stopping
        if diff_val_rs < self.thresh:
            self.stop_wait += 1
            if self.stop_wait > self.stop_patience:
                print('EARLY STOP')
                self.stop_wait = 0
                self.model.stop_training = True
        else: 
            self.stop_wait = 0

        # Case: train ratio isn't improving -> case for changing learn rate
        if diff_train_rs < self.thresh:
            self.lr_wait += 1
            if self.lr_wait > self.lr_patience:
                self.lr_wait = 0
                curr_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                self.model.optimizer.learning_rate.assign(curr_lr*0.5)
                print(f"Change LR -- Old: {curr_lr}, new: {curr_lr*0.5}")
        else:
            self.lr_wait = 0

        # Updating previous records
        self.prev_val_rs = val_rs
        self.prev_train_rs = train_rs

        # Epoch details
        avg_stats = (
            f"Avg Stats {epoch + 1} --"
            f" avg val RS: {round(self.sum_val_rs/(epoch + 1),5)},"
            f" avg train RS: {round(self.sum_train_rs/(epoch + 1),5)},"
            f" avg val RS change: {round(self.sum_diff_val_rs/(epoch),5)},"
            f" avg train RS change: {round(self.sum_diff_train_rs/(epoch),5)},"
            f" epoch: {epoch + 1}"
        )

        lr_stop_stats = (
            f"LR/Stop Stats {epoch + 1} --"
            f" valset diff: {round(diff_val_rs,5)},"
            f" stop counter: {self.stop_wait}/{self.stop_patience},"
            f" trainset diff: {round(diff_train_rs,5)},"
            f" lr counter: {self.lr_wait}/{self.lr_patience},"
        )
        print(avg_stats)
        print(lr_stop_stats)

        # Appending results to log
        with open(self.train_details_path, 'a') as file:
            file.write(f'{avg_stats}\n')


#########################################################
    # Custom Metrics #
#### Creating custom RatioSharpe Metric class
class RatioSharpe(tf.keras.metrics.Metric):
    def __init__(self, dtype = None, name = 'RS'):
        super().__init__(dtype, name)
        self.batch_rs = self.add_weight(
            name='batch_rs', 
            initializer='zeros',
            dtype=tf.float32
        )
        self.count = self.add_weight(
            name='count_batch_iterations', 
            initializer='zeros',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight = None):
        # Method is invoked at end of EACH batch
        batch_stock_ret_daily = y_pred * y_true
            # Shape: (batch_size, num_stocks)
        batch_port_ret_daily = tf.reduce_sum(batch_stock_ret_daily, axis=1)
            # Shape: (batch_size,)
        batch_rets_mean = tf.reduce_mean(batch_port_ret_daily)
            # Shape: (,)
        batch_rets_std = tf.math.reduce_std(batch_port_ret_daily)
            # Shape: (,)
        batch_rets_rs = batch_rets_mean/(batch_rets_std + tf.keras.backend.epsilon())

        # Update
        self.batch_rs.assign_add(batch_rets_rs)
        self.count.assign_add(1)

    def result(self):
        # Method is only executed at end of trainset/valset
        return self.batch_rs/self.count

    def reset_state(self):
        # Method is invoked at END of training part and validation/test part
            # After "result"
        self.batch_rs.assign(0.0)
        self.count.assign(0.0)



#########################################################
    # Plotting training history #
#### Path for images
developed_path = os.path.join(
    os.getcwd(),
    'DevelopedModels'
)
#### Function
def plot_history(history, model_num: int, fullset: str, strat_type: str, indicators: list[str]):
    fig, ax = plt.subplots(figsize=(12,5))
    # Train RS
    ax.plot(
        history.history['RS'],
        label = 'Training RS',
        color = 'blue'
    )
    # Val RS
    ax.plot(
        history.history['val_RS'],
        label='Val RS',
        color = 'orange'
    )
    ax.set_title(f'{fullset} - Strat: {strat_type}, Indicators: {", ".join(indicators)}')
    ax.set_ylabel('Ratio Sharpe Value')
    ax.set_xlabel('Epoch')
    ax.legend()

    # Guardando imagen
    plt.savefig(f'{developed_path}/Model{model_num}/Plots/{fullset}_{strat_type}_{"_".join(indicators)}.png', dpi=300)
    plt.close()


#########################################################
    # Training function #
##### Function
def train(
    data: dict, 
    indicators: list[str], 
    model_num: int, 
    strat: str,
    fs_start: str = 'fullset0'
):
    # Creating model
    model = LSTMModel(
        num_indicators = len(indicators),
        num_assets = NUM_STOCKS,
        name=f'{strat}_{len(indicators)}_indicators'
    )
    # Creating paths
    model_path = f'{developed_path}/Model{model_num}'
    os.makedirs(model_path, exist_ok=True)  # General Usage
    os.makedirs(f'{model_path}/Plots', exist_ok=True)    # For plots
    os.makedirs(f'{model_path}/TrainDetails', exist_ok=True)    # For training details
    # Creating resulting pandas
    df_weight_results = pd.DataFrame(
        columns = [f'Weight_{stock}' for stock in STOCK_NAMES] \
            + ['daily_ret']
    )
    df_weight_results.index = pd.to_datetime(df_weight_results.index)
    # Training 
    started_flag = False
    for FS_name, FS_train_val_test_list in data.items():
        # Choosing starting point
        FS_only_name = FS_name.split("_")[0]
        if (FS_only_name != fs_start) and (not started_flag): continue
        started_flag = True
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
            loss=MinRS,
            metrics=[RatioSharpe()]
        )
        # Setup data for current fullset
        trainset = FS_train_val_test_list[0]
        valset = FS_train_val_test_list[1]
        testset = FS_train_val_test_list[2]
        # Callback
        custom_cb = CustomCallback(train_details_path=f'{model_path}/TrainDetails/{FS_only_name}_train_details.txt')
        # Training
        history = model.fit(
            x=trainset[0],    # All windows' inputs
            y=trainset[1],    # All windows' labels
            batch_size=32,
            epochs=500,
            verbose=2,
            callbacks=custom_cb,
            validation_data=(valset[0], valset[1]),
            shuffle=False,
        )
        # Interrupt
        if global_stop_training: 
            # Save all previous info
            # Up to, but not including, the current fullset
            df_weight_results.to_csv(
                f'{model_path}/INCOMPLETE_weight_results.csv',
                index=True,
                encoding='utf-8'
            )
            break
        # Recording averages of curr fullset
        with open(f'{model_path}/TrainDetails/{FS_only_name}_train_details.txt', 'r') as origin,\
            open(f'{model_path}/TrainDetails/all_fs_avgs.txt', 'a') as dest:
            last_avg = origin.readlines()[-1]
            last_avg_info = last_avg.split(" -- ")[1]
            fs_avg = FS_only_name + " -- " + last_avg_info
            dest.write(fs_avg)
        # Graphing history
        plot_history(
            history, 
            model_num=model_num, 
            fullset=FS_only_name.capitalize(), 
            strat_type=strat, 
            indicators=indicators
        )
        # Testing model
        y_pred = model.predict(testset[0])
        daily_rets = np.sum(y_pred * testset[1], axis=1).reshape(-1,1)
            # Reshape: (num_windows,) -> (num_windows,1)
        full_results = np.concatenate([y_pred, daily_rets], axis=1)
        # Recording test results
        df_curr_fs_results = pd.DataFrame(
            full_results, 
            columns=[f'Weight_{stock}' for stock in STOCK_NAMES] + ['daily_ret'],
            index=testset[2]
        )
        df_weight_results = pd.concat([
            df_weight_results,
            df_curr_fs_results
        ], axis=0)
    # Saving all fullsets' completed results
    if FS_only_name == 'fullset23':
        df_weight_results.to_csv(
            f'{model_path}/weight_results.csv',
            index=True,
            encoding='utf-8'
        )

#########################################################
    # Training - Sliding Technique #
##### Stopping hotkey
global_stop_training = False
def hotkey():
    global global_stop_training
    global_stop_training = True
    return
keyboard.add_hotkey('ctrl+e', hotkey)

##### Model 1 - Sliding - Indicators: Price and Log returns
#### Setup corresponding data
indicators = ['Price', 'LogRet']
dict_fullsets_P_L_sliding = \
    indicator_selection(indicators, dict_sliding_fullsets)

#### Train Model
train(
    data = dict_fullsets_P_L_sliding, 
    indicators = indicators, 
    model_num = 1, 
    strat = 'Sliding',
)

#####  Model 2 - Sliding - Indicators: HLC3, TEMA, OBV
#### Setup corresponding data
indicators = ['TEMA', 'HLC3', 'OBV']
dict_fullsets_H_T_O_sliding = \
    indicator_selection(indicators, dict_sliding_fullsets)

#### Train model
train(
    data = dict_fullsets_H_T_O_sliding, 
    indicators = indicators, 
    model_num = 2, 
    strat = 'Sliding',
)

#####  Model 3 - Sliding - Indicators: All 5 
#### Setup corresponding data
indicators = ['Price', 'LogRet', 'TEMA', 'HLC3', 'OBV']
dict_fullsets_all_sliding = dict_sliding_fullsets

#### Train model
train(
    data = dict_fullsets_all_sliding, 
    indicators = indicators, 
    model_num = 3, 
    strat = 'Sliding',
)

#########################################################
    # Training - Expanding Technique #

#####  Model 4 - Expanding - Indicators: Price, LogRet
#### Setup corresponding data
indicators = ['Price', 'LogRet']
dict_fullsets_P_L_expand = \
    indicator_selection(indicators, dict_expanding_fullsets)

#### Train model
train(
    data = dict_fullsets_P_L_expand, 
    indicators = indicators, 
    model_num = 4, 
    strat = 'Expand',
)


#####  Model 5 - Expanding - Indicators: 'TEMA', 'HLC3', 'OBV'
#### Setup corresponding data
indicators = ['TEMA', 'HLC3', 'OBV']
dict_fullsets_H_T_O_expand = \
    indicator_selection(indicators, dict_expanding_fullsets)

#### Train model
train(
    data = dict_fullsets_H_T_O_expand, 
    indicators = indicators, 
    model_num = 5, 
    strat = 'Expand',
)


#####  Model 6 - Expanding - Indicators: all 5
#### Setup corresponding data
indicators = ['Price', 'LogRet', 'TEMA', 'HLC3', 'OBV']
dict_fullsets_all_expand = dict_expanding_fullsets

#### Train model
train(
    data = dict_fullsets_all_expand, 
    indicators = indicators, 
    model_num = 6, 
    strat = 'Expand',
)
