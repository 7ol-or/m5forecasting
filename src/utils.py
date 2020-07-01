import pandas as pd
import numpy as np
import os, sys, gc, time, warnings, pickle, psutil, random, json, re
from multiprocessing import Pool


__all__ = [
    'get_memory_usage',
    'sizeof_fmt',
    'reduce_mem_usage',
    'categorize',
    'dict_to_json',
    'json_prettify',
    'pickle_load',
    'pickle_dump',
    'merge_by_concat', 
    'df_parallelize_run',
    'duplicate_find'
]

def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def reduce_mem_usage(df, verbose=True):
    ## Memory Reducer
    # :df pandas dataframe to reduce size             # type: pd.DataFrame()
    # :verbose                                        # type: bool
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def categorize(df):
    # Convert "Object" type columns to "Category" type columns
    # The categorical data type is useful in some cases
    # https://stackoverflow.com/questions/30601830/when-to-use-category-rather-than-object
    for col in df.columns:
        if df[col].dtype.name == 'object':
            df[col] = df[col].astype('category')
    return df

def dict_to_json(obj, filepath):
    with open(filepath,'w') as fw:
        json.dump(obj, fw, indent=4, separators=(',', ': '))

def json_prettify(filepath):
    with open(filepath, 'r') as fr:
        jsn = fr.readlines()
        jsn = ''.join(jsn)
        jsn = re.sub(r'},','}@',jsn)
        jsn = re.sub(r'],',']@',jsn)
        jsn = re.sub(r',\n\s+',', ',jsn)
        jsn = re.sub(r'@',',',jsn)
    with open(filepath, 'w') as fw:
        fw.write(jsn)

def pickle_load(filepath):
    with open(filepath, 'rb') as fr:
        return pickle.load(fr)

def pickle_dump(obj, filepath):    
    with open(filepath, 'wb') as fw:        
        print('pickle saved.', filepath)
        return pickle.dump(obj, fw, pickle.HIGHEST_PROTOCOL)

def df_parallelize_run(func, t_split, n_cores):
    num_cores = np.min([n_cores,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df

def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1

def duplicate_find(l):
    return list(set([x for x in l if l.count(x) > 1]))