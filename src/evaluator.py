import os
import datetime
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from config import FLAGS
from utils import pickle_dump, pickle_load

EVAL_CACHE_FILE = None

_initialized = False

sw_df = None
roll_mat_csr = None
roll_index = None
y_true = None
top_idx = None
bottom_idx = None
top_y_true = None

D_START = 0
D_END = 0

def _cache_check():
    return os.path.exists(EVAL_CACHE_FILE)

def _save_cache():
    global sw_df, roll_mat_csr, roll_index, y_true, D_START, D_END
    pickle_dump((sw_df, roll_mat_csr, roll_index, y_true,  D_START, D_END), EVAL_CACHE_FILE)

def _load_cache():
    global sw_df, roll_mat_csr, roll_index, y_true, D_START, D_END, _initialized
    sw_df, roll_mat_csr, roll_index, y_true, D_START, D_END = pickle_load(EVAL_CACHE_FILE)
    _initialized = True

def _set_d(start, end):
    global D_START, D_END
    D_START = start
    D_END = end

def get_rollup_matrix(sales_df):
    dummies_list = [sales_df.state_id, sales_df.store_id, 
                    sales_df.cat_id, sales_df.dept_id, 
                    sales_df.state_id +'_'+ sales_df.cat_id, sales_df.state_id +'_'+ sales_df.dept_id,
                    sales_df.store_id +'_'+ sales_df.cat_id, sales_df.store_id +'_'+ sales_df.dept_id, 
                    sales_df.item_id, sales_df.state_id +'_'+ sales_df.item_id]
    dummies_df_list = [pd.DataFrame(np.ones(sales_df.shape[0]).astype(np.int8), 
                                    index=sales_df.index, columns=['all']).T]
    for cats in dummies_list:
        dummies_df_list.append(pd.get_dummies(cats, drop_first=False, dtype=np.int8).T)
    dummies_df_list.append(pd.DataFrame(np.eye(sales_df.shape[0]), index=sales_df.id))
    roll_mat_df = pd.concat(dummies_df_list, keys=list(range(12)), 
                            names=['level','id'])
    roll_index = roll_mat_df.index
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    return roll_index, roll_mat_csr

def get_s(sales_df, roll_mat_csr):
    d_name = ['d_' + str(i) for i in range(1, D_START)]
    sales_train_val = roll_mat_csr * sales_df[d_name].values
    no_sales = np.cumsum(sales_train_val, axis=1) == 0
    sales_train_val = np.where(no_sales, np.nan, sales_train_val)
    s = np.nanmean(np.diff(sales_train_val,axis=1)**2,axis=1)
    return s

def get_w(sales_df, calendar_df, prices_df):
    cols = ["d_{}".format(i) for i in range(D_START-28, D_START)]
    df_last28 = sales_df[["id", 'store_id', 'item_id'] + cols]
    df_last28 = df_last28.melt(id_vars=["id", 'store_id', 'item_id'], 
                    var_name="d", value_name="sale")
    df_last28 = pd.merge(df_last28, calendar_df, how = 'inner', 
                    left_on = ['d'], right_on = ['d'])
    df_last28 = df_last28[["id", 'store_id', 'item_id', "sale", "d", "wm_yr_wk"]]
    df_last28 = df_last28.merge(prices_df, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
    df_last28.drop(columns = ['wm_yr_wk'], inplace=True)
    df_last28['sale_usd'] = df_last28['sale'] * df_last28['sell_price']

    sale_usd = df_last28[['id','sale_usd']]
    sale_usd = sale_usd[sale_usd['id'].isin(sales_df['id'])]
    total_sales_usd = sale_usd.groupby(
        ['id'], sort=False)['sale_usd'].apply(np.sum).values
    print(roll_mat_csr.shape, total_sales_usd.shape)
    w = roll_mat_csr * total_sales_usd
    return 12*w/np.sum(w)

def _post_init():
    global top_idx, bottom_idx, top_y_true
    top_idx = [_id for level, _id in roll_index.values if level!=11]   
    bottom_idx = [_id for level, _id in roll_index.values if level==11]
    top_y_true = roll_mat_csr.dot(y_true)[:len(top_idx)]
    print('[evaluator] evaluate from d_{} to d_{}'.format(D_START, D_END))

def _init_evaluator(from_pickle=True, d_start=1914 ,d_end=1942):
    global EVAL_CACHE_FILE, sw_df, roll_mat_csr, roll_index, _initialized, y_true

    EVAL_CACHE_FILE = os.path.join(FLAGS.ARTIFACTS_PATH, f'EVAL_CACHE_FILE_{d_start}_{d_end}.pkl')
    _set_d(d_start, d_end)

    if from_pickle:
        try:
            _load_cache()
            _post_init()
            return             
        except:
            print('[evaluator] pickle load failed.')
            pass

    sales_df = pd.read_csv(os.path.join(FLAGS.DATA_PATH, 'sales_train_evaluation.csv'))
    y_true = sales_df[['d_'+str(_) for _ in range(D_START, D_END)]]
    calendar_df = pd.read_csv(os.path.join(FLAGS.DATA_PATH, 'calendar.csv'))
    prices_df = pd.read_csv(os.path.join(FLAGS.DATA_PATH, 'sell_prices.csv'))
    
    roll_index, roll_mat_csr = get_rollup_matrix(sales_df)    
    S = get_s(sales_df, roll_mat_csr)
    W = get_w(sales_df, calendar_df, prices_df)
    SW = W/np.sqrt(S)    
    sw_df = pd.DataFrame(np.stack((S, W, SW), axis=-1),index = roll_index,columns=['s','w','sw'])    
    _initialized = True
    _save_cache()
    _post_init()

def wrmsse(preds, return_df=False):
    '''
    preds - Predictions: Numpy or pd.DataFrame of size (M rows, N day columns)
    y_true - True values: Numpy or pd.DataFrame of size (M rows, N day columns)
    ''' 
    if isinstance(preds,pd.DataFrame):
        preds = preds.values
    s = sw_df.s.values
    w = sw_df.w.values
    score_matrix = (np.square(roll_mat_csr.dot(preds-y_true)) * np.square(w)[:, None])/ s[:, None]
    score_matrix[np.isnan(score_matrix)] = 0
    score_matrix = np.where(score_matrix==np.inf, 0, score_matrix)
    score_matrix = np.where(score_matrix==-np.inf, 0, score_matrix)
    if return_df:
        return np.sum(np.sqrt(np.mean(score_matrix,axis=1)))/12, pd.DataFrame(data=score_matrix, index=roll_index)
    else: 
        return np.sum(np.sqrt(np.mean(score_matrix,axis=1)))/12

def get_bottom(df, id_suffix='evaluation'):
    _df = df[df['id'].str.endswith(id_suffix)].copy()
    _df.loc[:,'id'] = _df['id'].str.replace('validation', 'evaluation')
    return _df.set_index('id').loc[bottom_idx]

def get_top(df, id_suffix='evaluation'):
    _df = df[df['id'].str.endswith(id_suffix)].copy()
    _df.loc[:,'id'] = _df['id'].str.replace('validation', 'evaluation')
    _df = _df.set_index('id').loc[bottom_idx]    
    _df = pd.DataFrame(
            data=roll_mat_csr.dot(_df.values),
            index=top_idx + bottom_idx,
            columns=[f'F{i}' for i in range(1,29)]
        )
    return _df.loc[top_idx]

def get_individual_loss(loss_df):
    '''
    ex)
    score, loss_df = evaluator.wrmsse(preds, return_df=True)
    evaluator.get_individual_loss(loss_df)
    '''
    return np.sqrt(np.mean(loss_df,axis=1))
    
def eval_submission(filepath, id_suffix='evaluation', return_df=False): 
    submission = pd.read_csv(filepath)
    preds = get_bottom(submission, id_suffix)
    return wrmsse(preds, return_df=return_df)

if __name__ == '__main__':
    start = datetime.datetime.now()
    _init_evaluator(from_pickle=False)
    end = datetime.datetime.now()
    print('Execution time', end - start)
else:
    _init_evaluator(from_pickle=True)