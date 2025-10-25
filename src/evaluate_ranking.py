import numpy as np
from sklearn.metrics import ndcg_score


def precision(site_pred, site_true, k):
    count = 0
    for s in site_pred[:k]:
        if s in site_true[:k]:
            count += 1
    return count / k


def recall(site_pred, site_true, k):
    count = 0
    for s in site_true[:k]:
        if s in site_pred[:k]:
            count += 1
    return count / len(site_true)


def evaluate_precision(result, top_dict):
    precision1_list = []
    precision3_list = []
    precision5_list = []
    precision10_list = []
    precision20_list = []
    precision30_list = []
    precision50_list = []

    result_new = {}
    for id1, id2_dict in result.items():
        if id1 in top_dict:
            id2_dict_new = {}
            for id2, score in id2_dict.items():
                if id2 in top_dict[id1]:
                    id2_dict_new[id2] = score
            
            id2_dict_new = {k: v for k, v in sorted(id2_dict_new.items(), key=lambda x: x[1], reverse=True)}

            result_new[id1] = id2_dict_new
            source_true = list(top_dict[id1].keys())
            source_pred = list(id2_dict_new.keys())

            if len(source_true) >= 1:
                precision1_list.append(precision(source_pred, source_true, 1))
            if len(source_true) >= 5:
                precision5_list.append(precision(source_pred, source_true, 5))
            if len(source_true) >= 10:
                precision10_list.append(precision(source_pred, source_true, 10))
            if len(source_true) >= 30:
                precision30_list.append(precision(source_pred, source_true, 30))
            if len(source_true) >= 50:
                precision50_list.append(precision(source_pred, source_true, 50))

    return {
            'Precision@1': np.mean(precision1_list),
            'Precision@5': np.mean(precision5_list),
            'Precision@10': np.mean(precision10_list),
            'Precision@30': np.mean(precision30_list),
            'Precision@50': np.mean(precision50_list),
           }
    

def evaluate_ndcg(result, top_dict):
    ndcg1_list = []
    ndcg3_list = []
    ndcg5_list = []
    ndcg10_list = []
    ndcg20_list = []
    ndcg30_list = []
    ndcg50_list = []

    result_new = {}
    for id1, id2_dict in result.items():
        if id1 in top_dict:
            id2_dict_new = {}
            for id2, score in id2_dict.items():
                if id2 in top_dict[id1]:
                    id2_dict_new[id2] = score
                    
            id2_dict_new_sorted = {k: v for k, v in sorted(id2_dict_new.items(), key=lambda x: x[1], reverse=True)}
            
            score_pred = []
            for id2_pred, _ in id2_dict_new_sorted.items():
                score_pred.append(top_dict[id1][id2_pred])
            score_true = list(top_dict[id1].values())
            
            score_pred = np.asarray([score_pred])
            score_true = np.asarray([score_true])

            

            try:
                ndcg1_list.append(ndcg_score(score_pred, score_true, k=1))
            except:
                from pdb import set_trace
                set_trace()
            ndcg5_list.append(ndcg_score(score_pred, score_true, k=5))
            ndcg10_list.append(ndcg_score(score_pred, score_true, k=10))
            ndcg30_list.append(ndcg_score(score_pred, score_true, k=30))
            ndcg50_list.append(ndcg_score(score_pred, score_true, k=50))

    return {
        'nDCG@1': np.mean(ndcg1_list),
        'nDCG@5': np.mean(ndcg5_list),
        'nDCG@10': np.mean(ndcg10_list),
        'nDCG@30': np.mean(ndcg30_list),
        'nDCG@50': np.mean(ndcg50_list),
       }