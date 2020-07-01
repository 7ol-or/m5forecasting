import os
import datetime
import time
import math
from itertools import product
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import FLAGS

DSTART = 1100 # to save memory 
SHIFTMEAN = list(product([1,3,5,7,10,14,21,28],[1,3,5,7,10,14,21,28]))
SHIFTMEAN = [(s, m) for s, m in SHIFTMEAN if m != 1]
SHIFTMEAN = [(s, m) for s, m in SHIFTMEAN if (s,m) not in [(7,7),(7,28),(28,7)]]
TE = FLAGS.TRAIN_END
PROCESSED_DATA_PATH = os.path.join(FLAGS.ARTIFACTS_PATH, "vertical_df.pkl")

vertical_df = None
horizon_shape = None

# SHIFTMEAN
# [(1, 3), (1, 5), (1, 7), (1, 10), (1, 14), (1, 21), (1, 28), (3, 3),
# (3, 5), (3, 7), (3, 10), (3, 14), (3, 21), (3, 28), (5, 3), (5, 5), 
# (5, 7), (5, 10), (5, 14), (5, 21), (5, 28), (7, 3), (7, 5), (7, 10), 
# (7, 14), (7, 21), (10, 3), (10, 5), (10, 7), (10, 10), (10, 14), (10, 21), 
# (10, 28), (14, 3), (14, 5), (14, 7), (14, 10), (14, 14), (14, 21), (14, 28), 
# (21, 3), (21, 5), (21, 7), (21, 10), (21, 14), (21, 21), (21, 28), (28, 3), 
# (28, 5), (28, 10), (28, 14), (28, 21), (28, 28)]

def create_lag_feature(colname, lagdays, default_value):
    '''
    ex. create_lag_feature('sales', 1, 0)
    '''
    global vertical_df, horizon_shape
    lag_array = np.roll(vertical_df[colname].values.astype(np.float32).reshape(horizon_shape), lagdays, axis=1)
    lag_array[:,:lagdays] = default_value
    return lag_array.flatten()

def create_rolling_feature(colname, shiftdays, rolldays, aggregate, default_value):
    '''
    ex. create_rolling_feature('sales', 1, 7, 'mean', np.nan)
    ex. create_rolling_feature('sales', 1, 7, 'sum', np.nan)
    '''
    global vertical_df, horizon_shape
    rolling_df = pd.DataFrame(vertical_df[colname].values.reshape(horizon_shape))
    rolling_array = rolling_df.T.shift(shiftdays).rolling(rolldays).agg(aggregate).T.values
    rolling_array[:,:rolldays] = default_value    
    return rolling_array.flatten().astype(np.float32)

def create_dynamic_features(_shift, _mean):
    '''
    ex. create_dynamic_features(1, 28)
    '''    
    global vertical_df    
    
    vertical_df['lag_sales_7'] = create_lag_feature('sales', 7, np.nan)
    vertical_df['lag_sales_28'] = create_lag_feature('sales', 28, np.nan)
    
    vertical_df['rolling_sales_mean_{}_{}'.format(_shift,_mean)] = create_rolling_feature('sales', _shift, _mean, 'mean', np.nan)
    
    for shift in [7, 28]: 
        for mean in [7, 28]:
            vertical_df['rolling_sales_mean_{}_{}'.format(shift,mean)] = create_rolling_feature('sales', shift, mean, 'mean', np.nan)              

def data_preprocessing():
    global vertical_df, horizon_shape

    train_df = pd.read_csv(os.path.join(FLAGS.DATA_PATH, 'sales_train_evaluation.csv'))

    prices_df = pd.read_csv(
        os.path.join(FLAGS.DATA_PATH, 'sell_prices.csv'), 
        dtype={
            "wm_yr_wk": "int16",
            "sell_price": "float32"
        })

    calendar_df = pd.read_csv(
        os.path.join(FLAGS.DATA_PATH, 'calendar.csv'),
        dtype={
            'wm_yr_wk': 'int16',
            "wday": "int16",
            "month": "int16",
            "year": "int16", 
            "snap_CA": "float32", 
            'snap_TX': 'float32', 
            'snap_WI': 'float32'
        }
    )

    id_cols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    d_cols = [col for col in train_df.columns if col.startswith('d_')][DSTART-1:]
    for col in id_cols:
        if col != 'id':
            train_df[col] = train_df[col].astype('category')
            train_df[col] = train_df[col].cat.codes.astype("int16")
            train_df[col] -= train_df[col].min()
    train_df = train_df.loc[:,id_cols+d_cols]
    for day in range(1942, 1942+28):
        train_df["d_{}".format(day)] = np.nan     

    for col in ['store_id', 'item_id']:
        prices_df[col] = prices_df[col].astype('category')
        prices_df[col] = prices_df[col].cat.codes.astype('int16')
        prices_df[col] -= prices_df[col].min()        

    for col in ['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2', 'weekday']:
        calendar_df[col] = calendar_df[col].astype('category')
        calendar_df[col] = calendar_df[col].cat.codes.astype('int16')
        calendar_df[col] -= calendar_df[col].min()

    vertical_df = pd.melt(
        train_df, 
        id_vars=id_cols,
        value_vars=[d for d in train_df.columns if d.startswith('d_')],
        var_name="d",
        value_name="sales")
    vertical_df = vertical_df.merge(calendar_df, on="d", copy=False)
    vertical_df = vertical_df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how='left', copy=False)

    vertical_df['d'] = vertical_df['d'].apply(lambda x:x[2:]).astype(np.int16)
    vertical_df['date'] = pd.to_datetime(vertical_df['date'])
    vertical_df['tm_d'] = vertical_df['date'].dt.day.astype(np.int8)
    vertical_df['tm_w'] = vertical_df['date'].dt.week.astype(np.int8)
    vertical_df['tm_m'] = vertical_df['date'].dt.month.astype(np.int8)
    vertical_df['tm_q'] = vertical_df['date'].dt.quarter.astype(np.int8)
    vertical_df['tm_y'] = vertical_df['date'].dt.year.astype(np.int16)
    vertical_df['tm_y'] = (vertical_df['tm_y'] - vertical_df['tm_y'].min()).astype(np.int8)
    vertical_df['tm_wm'] = vertical_df['tm_d'].apply(lambda x: math.ceil(x/7)).astype(np.int8)
    vertical_df['tm_dw'] = vertical_df['date'].dt.dayofweek.astype(np.int8)
    vertical_df['tm_w_end'] = (vertical_df['tm_dw']>=5).astype(np.int8)

    vertical_df = vertical_df.reset_index().sort_values(['id','d'])
    horizon_shape = train_df[[d for d in train_df.columns if d.startswith('d_')]].shape

    # non dynamic features
    vertical_df['lag_sales_28'] = create_lag_feature('sales', 28, np.nan)
    vertical_df['rolling_sales_mean_28_28'] = create_rolling_feature('sales', 28, 28, 'mean', np.nan)
    vertical_df['rolling_sales_mean_28_14'] = create_rolling_feature('sales', 28, 14, 'mean', np.nan)
    vertical_df['rolling_sales_mean_28_7'] = create_rolling_feature('sales', 28, 7, 'mean', np.nan)
    vertical_df['rolling_sales_std_28_28'] = create_rolling_feature('sales', 28, 28, 'std', np.nan)
    vertical_df['rolling_sales_std_28_14'] = create_rolling_feature('sales', 28, 14, 'std', np.nan)
    vertical_df['rolling_sales_std_28_7'] = create_rolling_feature('sales', 28, 7, 'std', np.nan)
    # vertical_df['rolling_isonsale_1'] = vertical_df['sell_price'] / create_lag_feature('sell_price', 1, np.nan)
    # vertical_df['rolling_isonsale_7'] = vertical_df['sell_price'] / create_rolling_feature('sell_price', 1, 7, 'mean', np.nan)

    vertical_df.to_pickle(PROCESSED_DATA_PATH)

def _run(_shift, _mean):
    global vertical_df
    
    print('_shift:', _shift, '_mean:',_mean)
    create_dynamic_features(_shift, _mean)
    
    cat_features = ['dept_id', 'store_id', 'cat_id', 'state_id', 'event_name_1','event_type_1','event_name_2','event_type_2']
    features = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'month', 'year',
                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 
                'lag_sales_7', 'lag_sales_28',
                'rolling_sales_mean_28_7','rolling_sales_mean_28_14','rolling_sales_mean_28_28',
                'rolling_sales_mean_7_7', 'rolling_sales_mean_7_28', 'tm_d']
    features += [f'rolling_sales_mean_{_shift}_{_mean}']
    
    trainset = vertical_df.dropna().query(f'{TE-FLAGS.TRAIN_DAYS}<=d<{TE}')[features]
    trainlabel = vertical_df.dropna().query(f'{TE-FLAGS.TRAIN_DAYS}<=d<{TE}')['sales']
    train_data = lgb.Dataset(
        trainset,
        label=trainlabel,
        categorical_feature=cat_features,
        free_raw_data=True)

    validset = vertical_df.dropna().query(f'{TE}<=d<{TE+28}')[features]
    validlabel = vertical_df.dropna().query(f'{TE}<=d<{TE+28}')['sales']
    valid_data = lgb.Dataset(
        validset,
        label=validlabel,
        categorical_feature=cat_features,
        free_raw_data=True)

    params = FLAGS.HPARAMS.copy()

    estimator = lgb.train(params,
                          train_data,
                          valid_sets=[train_data, valid_data],
                          verbose_eval=False
                          )

    estimator.save_model(os.path.join(FLAGS.ARTIFACTS_PATH, f"estimator_{TE}_{_shift}_{_mean}.lgb"))

    for day in range(TE, TE+28):
        input_df = vertical_df[vertical_df['d']==day][features]
        pred = estimator.predict(input_df)
        vertical_df.loc[vertical_df['d']==day,'sales'] = pred
        create_dynamic_features(_shift, _mean)

    submission = vertical_df.query(f'{TE}<=d<{TE+28}')[['id','d','sales']]
    submission = submission.pivot(index='id',columns='d',values='sales').reset_index()
    submission.columns = ['id']+[f'F{f}'for f in range(1,29)]
    submission['id'] = submission['id'].str.replace('evaluation', 'validation')
    submission.to_pickle(
        os.path.join(FLAGS.SUBMISSIONS_PATH, f'validation_{TE}_{_shift}_{_mean}.pkl')
        )

def run():
    global vertical_df    
    for _shift, _mean in SHIFTMEAN:    
        if not os.path.exists(PROCESSED_DATA_PATH):
            data_preprocessing()
        vertical_df = pd.read_pickle(PROCESSED_DATA_PATH)
        _run(_shift, _mean)