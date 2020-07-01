import pandas as pd
import numpy as np
from scipy import linalg
from scipy.sparse import diags, csr_matrix
import os
import glob
import evaluator
from config import FLAGS

VALIDATIONS = ['validation_1914','validation_1900', 'validation_1886', 'validation_1872', 'validation_1858']
EVALUATION = 'validation_1942'

def load_validation_files():
    global df, validation_subs

    validation_subs = []
    for df_name in VALIDATIONS:
        validation_subs.extend(glob.glob(os.path.join(FLAGS.SUBMISSIONS_PATH, df_name+'*')))
        globals()[df_name] = pd.DataFrame()
    validation_subs.sort()

    for sf in validation_subs:
        df_name = '_'.join(sf.split('_')[:2])
        feature_name = 'f_'+'_'.join(sf.split('_')[2:4])
        print(df_name, feature_name)
        feature = evaluator.get_bottom(pd.read_pickle(sf), 'validation').values.flatten().astype(np.float32)
        globals()[df_name][feature_name] = feature

    for df_name in set(['_'.join(sf.split('_')[:2]) for sf in validation_subs ]):    
        globals()[df_name]['day'] = np.tile(np.arange(1,29),30490)    
        train_end = int(df_name.split('_')[1])
        globals()[df_name]['train_end'] = train_end    
        evaluator._init_evaluator(True,train_end,train_end+27)    
        sw = np.repeat(evaluator.sw_df.fillna(0).xs(11).sw.values, 28)
        globals()[df_name]['sw'] = sw    
        y_true = evaluator.y_true.values.flatten().astype(np.float32)    
        globals()[df_name]['y_true'] = y_true

    df = pd.concat([globals()[df_name] for df_name in set(['_'.join(sf.split('_')[:2]) for sf in validation_subs ])], 
                axis = 0,
                ignore_index=True)

def _get_g_weight(g_param):
    res = []
    for gw in np.array(g_weight).T:
        res.append(gw**g_param / (gw**g_param).sum())
    return np.expand_dims(np.array(res).T, -1)

def _get_d_weight(d_param):
    res = []
    for dw in np.array(d_weight).T:
        res.append(dw**d_param / (dw**d_param).sum())
    return np.expand_dims(np.array(res).T, 1)

def ensemble_weight_grid_search(train_end):
    global g_weight, d_weight

    groupscore = []
    dayscore = []
    for df_name in VALIDATIONS:
        te = int(df_name.split('_')[1])
        if te >= train_end:
            continue
        evaluator._init_evaluator(True, te, te+27)
        grouploss_list = []
        dayloss_list = []
        for fea in [col for col in df.columns if col.startswith('f_')]:
            loss_df = evaluator.wrmsse(globals()[df_name][fea].values.reshape(30490,28), return_df=True)[1]
            grouploss = evaluator.get_individual_loss(loss_df).reset_index().groupby('level').sum().values.flatten()
            dayloss = loss_df.sum().values
            grouploss_list.append(grouploss)
            dayloss_list.append(dayloss)
        groupscore.append(grouploss_list)
        dayscore.append(dayloss_list)
        
    groupscore = np.array(groupscore)
    dayscore = np.array(dayscore)

    g_weight = 1 / groupscore.sum(0)
    d_weight = 1 / dayscore.sum(0)
    g_weight = (g_weight / g_weight.sum(0))
    d_weight = (d_weight / d_weight.sum(0))

    evaluator._init_evaluator(True, train_end, train_end+27)    
    group_count = evaluator.sw_df.groupby('level')['s'].count().values
    _g_weight_rollup = []
    for gw in g_weight:
        _g_weight_rollup.append(np.repeat(gw, group_count))
    g_weight = np.array(_g_weight_rollup)

    valid_data = df.query(f'train_end == {train_end}')[[col for col in df.columns if col.startswith('f_')]]
    valid_data = valid_data.values.T.reshape(53,30490,28)
    valid_data_rollup = []
    for vd in valid_data:
        valid_data_rollup.append(evaluator.roll_mat_csr @ vd)
    valid_data_rollup = np.array(valid_data_rollup)

    param_space = np.hstack([np.linspace(0,0.9,4), np.linspace(1,10,10)])
    y_true_rollup = evaluator.roll_mat_csr @ evaluator.y_true    
    scores = []
    for g in param_space:  
        for d in param_space:        
            pred_rollup = valid_data_rollup * _get_g_weight(g) * 53 * _get_d_weight(d) 
            score_matrix = ((y_true_rollup - pred_rollup.sum(0))* evaluator.sw_df.sw.values.reshape(-1,1))**2
            score = np.sum(np.sqrt(np.mean(score_matrix,axis=1)))/12
            scores.append((g,d,score))
    globals()[f'score_df_{train_end}'] = pd.DataFrame(scores, columns = ['g','d','score']).set_index(['g','d'])

def find_optimal_ensemble_weight():
    for df_name in VALIDATIONS:
        train_end = int(df_name.split('_')[1])
        ensemble_weight_grid_search(train_end)
    scores_df = sum([globals()[f'score_df_{train_end}'] for train_end in VALIDATIONS])
    optimal_idx = scores_df.sort_values('score').index
    g = scores_df.loc[optimal_idx].reset_index()['g'].values
    d = scores_df.loc[optimal_idx].reset_index()['d'].values
    return g, d 

def inv(a):    
    l=np.linalg.cholesky(a)
    linv=linalg.solve_triangular(l,np.eye(a.shape[0]),lower=True)
    ainv=linv.T@linv
    return ainv

def reconcile(hat_mat,sum_mat):
    diag_mat_inv = diags(1/np.sum(sum_mat.toarray().astype('float32'), axis = 1).reshape(-1))
    new_mat = sum_mat@(inv((sum_mat.T@diag_mat_inv@sum_mat).toarray().astype('float32'))@(sum_mat.T@(diag_mat_inv@hat_mat)))
    return new_mat

def makesubmission(pred_rollup):
    submission = pd.DataFrame(
        data = pred_rollup[-30490:],
        columns = [f'F{i}' for i in range(1,29)],
    )
    submission['id'] = evaluator.bottom_idx
    submission = submission.set_index('id').reset_index()
    submission.to_csv(os.path.join(FLAGS.SUBMISSIONS_PATH, 'submission.csv'), index=False)

def run():
    load_validation_files()
    g, d = find_optimal_ensemble_weight()
    gw, dw = _get_g_weight(g), _get_d_weight(d)
    evaluation_sub = glob.glob(os.path.join(FLAGS.SUBMISSIONS_PATH, EVALUATION+'*'))
    evaluation_sub.sort()
    evaluation_df = pd.DataFrame()

    for es in evaluation_sub:
        feature_name = 'f_'+'_'.join(es.split('_')[2:4])
        feature = evaluator.get_bottom(pd.read_pickle(es), 'evaluation').values.flatten().astype(np.float32)
        evaluation_df[feature_name] = feature
    
    valid_data = evaluation_df.values.T.reshape(53,30490,28)
    valid_data_rollup = []
    for vd in valid_data:
        valid_data_rollup.append(evaluator.roll_mat_csr @ vd)
    valid_data_rollup = np.array(valid_data_rollup)

    pred_rollup = valid_data_rollup * gw * 53 * dw
    pred_rollup = pred_rollup.sum(0)
    pred_rollup = reconcile(pred_rollup, evaluator.roll_mat_csr)
    makesubmission(pred_rollup)