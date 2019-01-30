# -*- encoding=utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np


def get_score(wi_list,data):
    """
    :param wi_list: 权重系数列表
    :param data：评价指标数据框
    :return:返回得分
    """

    #  将权重转换为矩阵

    cof_var = np.mat(wi_list)

    #  将数据框转换为矩阵
    context_train_data = np.mat(data)

    #  权重跟自变量相乘
    last_hot_matrix = context_train_data * cof_var.T
    last_hot_matrix = pd.DataFrame(last_hot_matrix)

    #  累加求和得到总分
    last_hot_score = list(last_hot_matrix.apply(sum))

    #  max-min 归一化

    # last_hot_score_autoNorm = autoNorm(last_hot_score)

    # 值映射成分数（0-100分）

    # last_hot_score_result = [i * 100 for i in last_hot_score_autoNorm]

    return last_hot_score



def get_entropy_weight(data):
    """
    :param data: 评价指标数据框
    :return: 各指标权重列表
    """
    # 数据标准化
    data = (data - data.min())/(data.max() - data.min())
    m,n=data.shape
    #将dataframe格式转化为matrix格式
    data=data.as_matrix(columns=None)
    k=1/np.log(m)
    yij=data.sum(axis=0)
    #第二步，计算pij
    pij=data/yij
    test=pij*np.log(pij)
    test=np.nan_to_num(test)

    #计算每种指标的信息熵
    ej=-k*(test.sum(axis=0))
    #计算每种指标的权重
    wi=(1-ej)/np.sum(1-ej)

    wi_list=list(wi)


    return  wi_list



if __name__ == '__main__':


    data0 = pd.read_excel("C:\\Users\\Oreo\\Desktop\\test2.xlsx", encoding='utf8')

    data = data0.iloc[:, 1:10]
    mm=data
    wi_list=get_entropy_weight(data)
    score_list=get_score(mm,wi_list)
    mm['score']=score_list
    mm['科室']=data0['科室']
    # 然后对数据框按得分从大到小排序
    result = mm.sort_values(by='score', axis=0, ascending=False)
    result['rank'] = range(1, len(result) + 1)

    print(result)

    # 写出csv数据
    result.to_csv('C:\\Users\\Oreo\\Desktop\\test2_result.csv', index=False)
