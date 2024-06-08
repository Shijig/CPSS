
import numpy as np
from noise import generate_noise
import random
from tensorflow.keras.utils import to_categorical

def data_t(data):
    out=[]
    for i in range(len(data)):
        out.append(data[i].T)
    return np.array(out)



def data_add_noise(data):   #强变换
    out=data.copy()
    for j in range(12):
        out[:,j]=out[:,j]+np.squeeze(generate_noise(random.uniform(0.5,1.2)))
    return out
def set_random_segment_zeros(x,r):
# 计算需要设置为0的数据片段的长度
    segment_length = int(r * 2048)
    # 随机生成一个起始位置，使得片段不会超过数组的末尾
    start_pos = np.random.randint(2048 - segment_length + 1)
    # 将x中对应位置的元素设置为0
    x[start_pos:start_pos+segment_length] = 0
    return x

def augment(data, r=0.4):
    out=data.copy()
    for j in range(12):
        out[:,j]=set_random_segment_zeros(out[:,j],r)
    return out

class CLRGenerator(object):
    def __init__(self, batch_size,data_path,label_rate,num_classes):
        self.data_path = data_path
        self.batch_size = batch_size
        self.label_rate=label_rate
        self.num_classes=num_classes
        self.data = data_t(np.load(self.data_path+'X_train.npy'))
        self.label = to_categorical(np.load(self.data_path+'Y_train.npy'),num_classes=self.num_classes)
        # self.data_n=np.load("../data/SH_dataset/X_train_lead_n.npy")
        self.num_label=int(self.label_rate*self.batch_size)
        self.num_unlabel=self.batch_size-self.num_label
        self.patient_num = self.data.shape[0]
        self.patient_ids = np.arange(self.patient_num)

        self.segment_length_1d = self.data.shape[1]
        self.n_samples = self.data.shape[0]
        self.n_batches = self.n_samples // self.batch_size


    def get_batch(self):
        x = np.empty(
            (2 * self.num_unlabel+self.num_label, self.segment_length_1d, 12),
            dtype=np.float32,
        )
        y=np.empty(
            (2 * self.num_unlabel+self.num_label, self.num_classes),
            dtype=np.float32,
        )
        idx_a = np.arange(self.num_label)
        batch_patients = np.random.choice(self.patient_num, self.batch_size, replace=False)
        data_batch=self.data[batch_patients]
        label_batch=self.label[batch_patients]
        
        x[idx_a] = data_batch[idx_a]
        y[idx_a] = label_batch[idx_a]
        
        y[self.num_label:]=np.zeros([2 * self.num_unlabel, self.num_classes])
        for i in range(self.num_unlabel):
            x[i+self.num_label] = data_batch[i+self.num_label]
            # x[self.batch_size+i] = x[i]+generate_noise(random.uniform(0.35,2.2))  #-6-12DB
            x[self.batch_size+i] = augment(data_batch[i+self.num_label],0.2)
        #     # x[self.batch_size+i] = x[i]+generate_noise(1.5)
        # x[self.batch_size+idx_a] = self.data_n[batch_patients]
        return x, y

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

    def __next__(self):
        return self.get_batch()

if __name__ == '__main__':
    database='PTB_XL'
    data_path='data/'+database+"/"
    label_rate=0.05
    num_classes=5
    generator = CLRGenerator(256,data_path,label_rate,num_classes)
    batch_x, y = generator.get_batch()
    batch_x1, y1 = generator.get_batch()
    print(batch_x.shape)
    # import matplotlib.pyplot as plt
    # idx0 = 0
    # idx1 = int(len(batch_x)/2)
    # plt.subplot(2, 1, 1)
    # plt.plot(batch_x[idx0], "b")
    # plt.grid(True)
    # plt.subplot(2, 1, 2)
    # plt.plot(batch_x[idx1], "r")
    # plt.grid(True)
    # plt.show()
