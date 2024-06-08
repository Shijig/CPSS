# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:11:09 2023

@author: shijigang
"""

import numpy as np
from tensorflow.keras.utils import to_categorical
from collections import Counter

def data_t(data):
    out=[]
    for i in range(len(data)):
        out.append(data[i].T)
    return np.array(out)
def set_random_segment_zeros(x,r):
# 计算需要设置为0的数据片段的长度
    segment_length = int(r * 2048)
    start_pos = np.random.randint(2048 - segment_length + 1)
    x[start_pos:start_pos+segment_length] = 0
    return x

def augment(data, r=0.4):
    out=data.copy()
    for j in range(12):
        out[:,j]=set_random_segment_zeros(out[:,j],r)
    return out

def data_generate(database='chapman',label_rate=0.05):

    data_path='data/'+database+"/"
    
    data = data_t(np.load(data_path+'X_train.npy'))
    y_labels=np.load(data_path+'Y_train.npy')
    num_classes=len(Counter(y_labels))
    labels = to_categorical(y_labels,num_classes=num_classes)
    
    num_labeled=int(len(labels)*label_rate)
    index_labeled = np.random.choice(len(labels), num_labeled, replace=False)
    data_labeled=data[index_labeled]
    label_labeled=labels[index_labeled]
    
    data_unlabeled0=[data[i] for i in range(len(data)) if i not in index_labeled]
    label_unlabeled0=[labels[i] for i in range(len(labels)) if i not in index_labeled]
    shuffled_indices = np.arange(len(data_unlabeled0))
    np.random.shuffle(shuffled_indices)
    data_unlabeled_u1 = np.array([data_unlabeled0[i] for i in shuffled_indices])
    label_unlabeled = np.array([label_unlabeled0[i] for i in shuffled_indices])
    
    def counter_classes(unlabels):
        unlabeled_class=np.argmax(unlabels, axis=1)
        data_counts = Counter(unlabeled_class)
        # 获取按数据大小排序的数据和对应的数量
        sorted_data_counts = sorted(data_counts.items(), key=lambda x: x[0], reverse=False)
        # 生成一个数组存放数据从大到小对应的数量
        sorted_counts_array = np.array([count for _, count in sorted_data_counts])
        # sorted_counts_array = np.around(sorted_counts_array, decimals=2)
        return sorted_counts_array
    # print("unlabeled total : labeled total ",round((1-label_rate)/label_rate,2))
    # print("rate of each class",np.around(counter_classes(label_unlabeled)/counter_classes(label_labeled),decimals=2))
    x_test=data_t(np.load(data_path+'X_test.npy'))
    y_test=to_categorical(np.load(data_path+'Y_test.npy'),num_classes=num_classes)
    
    data_unlabeled_u2=[]
    for i in range(len(data_unlabeled_u1)):
        data_unlabeled_u2.append(augment(data_unlabeled_u1[i],0.2))
    data_unlabeled_u2=np.array(data_unlabeled_u2)
    return data_labeled,label_labeled,data_unlabeled_u1,data_unlabeled_u2,x_test,y_test,num_classes
