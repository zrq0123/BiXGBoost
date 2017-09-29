import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from scipy.stats import norm
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.tree.tree import BaseDecisionTree
import csv
import random


def single_col_timeseries(scol,split,timelag,maxlag):
    slen=int(len(scol)/split)
    res_col=pd.Series()
    for index in range(0,split):
        tmp=scol[int(index*slen+timelag):(slen+index*slen+timelag-maxlag)]
        res_col=res_col.append(tmp,ignore_index=True)
    return  res_col


#exp_mat is dataframe of pandas,split is use to split time-series different samples,timelag is lag of the time,numrand is the num of add random lists
def invert_expression_timeseries(exp_mat,split,maxlag,pan=0):
    df = pd.DataFrame()
    all_mean=np.mean(exp_mat.values)
    all_std=np.std(exp_mat.values)
    for index in range(0, len(exp_mat.columns)):
        sname=exp_mat.columns[index];
        #df[sname]=exp_mat[sname]
        for jindex in range(0+pan,maxlag+pan):
            #use random to change the sequence
            df[(sname+'_'+str(maxlag-jindex))]=single_col_timeseries(exp_mat[sname],split,jindex,maxlag)
    return df

def get_Goldset(file_path):
    gold_set=set()
    file = open(file_path)
    for line in file:
        tmp_str=line.split('\t')
        gold_set.add((tmp_str[0]+"\t"+tmp_str[1]))
    return  gold_set


def getlinks(target_name,name,importance_,gold_set):
    feature_imp=pd.DataFrame(importance_,index=name,columns=['imp'])
    #feature_imp_drop=feature_imp_drop.sort_values(by="imp", ascending=False)
    feature_large_set = {}
    for i in range(0,len(feature_imp.index)):
        tmp_name=feature_imp.index[i].split('_')
        if tmp_name[0] !=target_name:
            if (tmp_name[0]+"\t"+target_name) not in feature_large_set:
                g_f=0
                if (tmp_name[0]+"\t"+target_name) in gold_set:
                    g_f=1
                tf_score =feature_imp.loc[feature_imp.index[i], 'imp']
                #tf_score=np.exp(-e_k*int(tmp_name[1])*int(tmp_name[1]))*tf_score #用于多时间片下，之间衰减
                #tf_score =tf_score/maxlag
                feature_large_set[tmp_name[0]+"\t"+target_name]=[tf_score,g_f]
                #large_set.append([(tmp_name[0]+"\t"+target_name),tf_score,g_f]) #add by max importance
                #change for 2
            else:
                tf_score =feature_imp.loc[feature_imp.index[i], 'imp']
                #tf_score = np.exp(-e_k * int(tmp_name[1])*int(tmp_name[1])) * tf_score #用于多时间片下，之间衰减
                #tf_score = tf_score / maxlag
                #feature_large_set[tmp_name[0] + "\t" + target_name][0] += tf_score
                feature_large_set[tmp_name[0] + "\t" + target_name][0] = max(feature_large_set[tmp_name[0] + "\t" + target_name][0],tf_score)

    return feature_large_set

def getlink_re(target_name,name,importance_,gold_set):
    feature_imp=pd.DataFrame(importance_,index=name,columns=['imp'])
    #feature_imp_drop=feature_imp_drop.sort_values(by="imp", ascending=False)
    feature_large_set = {}
    for i in range(0,len(feature_imp.index)):
        tmp_name=feature_imp.index[i].split('_')
        if tmp_name[0] !=target_name:
            if (target_name+"\t"+tmp_name[0]) not in feature_large_set:
                g_f=0
                if (target_name+"\t"+tmp_name[0]) in gold_set:
                    g_f=1
                tf_score =feature_imp.loc[feature_imp.index[i], 'imp']
                feature_large_set[target_name+"\t"+tmp_name[0]]=[tf_score,g_f]
            else:
                tf_score =feature_imp.loc[feature_imp.index[i], 'imp']

                feature_large_set[target_name+"\t"+tmp_name[0]][0] = max(feature_large_set[ target_name+"\t"+tmp_name[0]][0],tf_score)

    return feature_large_set

def choose_important_feature(name,importance_):
    feature_imp = pd.DataFrame(importance_, index=name, columns=['imp'])
    # feature_imp_drop=feature_imp_drop.sort_values(by="imp", ascending=False)
    feature_large_set = {}
    max_feature={}
    for i in range(0, len(feature_imp.index)):
        tmp_name = feature_imp.index[i].split('_')
        if (tmp_name[0] ) not in feature_large_set:
            tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
            # tf_score=np.exp(-e_k*int(tmp_name[1])*int(tmp_name[1]))*tf_score #用于多时间片下，之间衰减
            # tf_score =tf_score/maxlag
            feature_large_set[tmp_name[0]] = tf_score
            # large_set.append([(tmp_name[0]+"\t"+target_name),tf_score,g_f]) #add by max importance
            # change for 2
            max_feature[tmp_name[0]]=feature_imp.index[i]
        else:
            tf_score = feature_imp.loc[feature_imp.index[i], 'imp']
            # tf_score = np.exp(-e_k * int(tmp_name[1])*int(tmp_name[1])) * tf_score #用于多时间片下，之间衰减
            # tf_score = tf_score / maxlag
            # feature_large_set[tmp_name[0] + "\t" + target_name][0] += tf_score
            if feature_large_set[tmp_name[0]]<tf_score:
                max_feature[tmp_name[0]] = feature_imp.index[i]
                feature_large_set[tmp_name[0]]=tf_score
    return list(max_feature.values())


def compute_feature_importances(score_1,score_2,dicts_1,dicts_2):
    """Computes variable importances from a trained tree-based model.
    """
    dict_all_1={}
    dict_all_2 = {}
    score_1=score_1/sum(score_1)
    score_2 = score_2 / sum(score_2)
    for i in range(len(score_1)):
        tmp_dict=dicts_1[i]
        for key in tmp_dict:
            tmp_dict[key][0]=tmp_dict[key][0]*score_1[i]
        dict_all_1.update(tmp_dict)

    for i in range(len(score_2)):
        tmp_dict=dicts_2[i]
        for key in tmp_dict:
            tmp_dict[key][0]=tmp_dict[key][0]*score_2[i]
        dict_all_2.update(tmp_dict)

    d1 = pd.DataFrame.from_dict(dict_all_1, orient='index')
    d1.columns=["score_1", "key_1"]
    d2 = pd.DataFrame.from_dict(dict_all_2, orient='index')
    d2.columns = ["score_2", "key_2"]

    all_df = d1.join(d2)
    all_df['total'] = np.sqrt(all_df["score_1"] * all_df["score_2"])

    return all_df


def mainRun(expressionFile,goldFile,samples,outputfile,p_lambda=0,p_alpha=1,maxlag=2,timelag=2):

    #tfs = np.genfromtxt('test2/tfs_1.txt',dtype=str,delimiter='\n')
    data=pd.read_csv(expressionFile,'\t')
    #data=data.loc[0:21,:]
    g_set=get_Goldset(goldFile)
    seed=random.randint(1, 100)
    score_1=[]
    score_2=[]
    dict_1=[]
    dict_2=[]

    #data=data.apply(lambda S:(S-np.mean(S))/np.std(S))

    for index in range(0, len(data.columns)):
        t_data=data.copy()
        y=single_col_timeseries(data[data.columns[index]],samples,2,2)
        #rarray = np.concatenate((np.random.uniform(0.5,1,9),np.random.uniform(0,0.5,int(len(y)/10)-9)))
        #rarray=np.tile(rarray,10)
        y_normal=(y-np.mean(y))/np.std(y)
        x_c=invert_expression_timeseries(t_data,samples,2)
        clfx = xgb.XGBRegressor(n_jobs=1,max_depth=3,n_estimators=1000,learning_rate=0.0001,subsample=0.8,reg_lambda=p_lambda,reg_alpha=p_alpha,colsample_bylevel=0.6,colsample_bytree=0.6,seed=seed)
        clfx.fit(x_c,y_normal)
        err_1=mean_squared_error(clfx.predict(x_c), y_normal)

        #print(clfx.evals_result(),err_1)
        _importance_per=clfx.feature_importances_
        tmp_large = getlinks(data.columns[index], x_c.columns.values, _importance_per, g_set)
        score_1.append(err_1)
        dict_1.append(tmp_large)


        t_data = data.copy()
        y = single_col_timeseries(data[data.columns[index]], samples, 0, 2)
        y_normal = (y - np.mean(y)) / np.std(y)
        x_c = invert_expression_timeseries(t_data, samples, 2,1)
        clf = xgb.XGBRegressor(n_jobs=1,max_depth=3,n_estimators=1000,learning_rate=0.0001,subsample=0.8,reg_lambda=p_lambda,reg_alpha=p_alpha,colsample_bylevel=0.6,colsample_bytree=0.6,seed=seed)
        clf.fit(x_c, y_normal)
        err_2=mean_squared_error(clf.predict(x_c), y_normal)
        _importance_per_re = clf.feature_importances_
        tmp_small = getlink_re(data.columns[index], x_c.columns.values, _importance_per_re, g_set)
        score_2.append(err_2)
        dict_2.append(tmp_small)

        print("run------"+str(index))

    all_df=compute_feature_importances(score_1,score_2,dict_1,dict_2)
    #all_df.to_csv("test4/test_large_1_100_1.csv", sep="\t", header=False,quoting=csv.QUOTE_NONE,escapechar=" ")
    all_df[['total','key_1']].to_csv(outputfile, sep="\t", header=False,quoting=csv.QUOTE_NONE,escapechar=" ")



