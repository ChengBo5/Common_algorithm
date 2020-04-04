import tensorflow as tf
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv("numeric_feature.csv") # 读取数据

def feature_label_split(pd_data):
    #行数、列数
    row_cnt = pd_data.shape[0]
    column_cnt = pd_data.shape[1]
    #生成新的X、Y矩阵
    train_X = np.empty([row_cnt, column_cnt-2])   #生成两个随机未初始化的矩阵
    train_label = np.empty([row_cnt, 1])

    train_X = np.array(pd_data.ix[:,2:-1])
    label = np.array(pd_data.ix[:,1])
    #转为one-hot encoding编码    0:(1,0)   1:(0,1)
    train_label = np.array(pd.get_dummies(label, drop_first=False))

    return train_X, train_label

#把特征数据进行标准化为均匀分布
def uniform_norm(X_in):
    X_max = X_in.max(axis=0)
    X_min = X_in.min(axis=0)
    X = (X_in-X_min)/(X_max-X_min)
    return X, X_max, X_min

train_X , train_label = feature_label_split(train_data)
train_X , _ , _ =uniform_norm(train_X)  #数据归一化处理

batch_size = train_X.shape[0]
column_x = train_X.shape[1]
class_num = train_label.shape[1]
# print(train_label.head(10))
# print('batch_size:',batch_size,'column_x:',column_x,'class_num:',class_num)




sess = tf.Session()
# 加载模型
saver = tf.train.import_meta_graph('models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('models'))
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]

graph = tf.get_default_graph()
x_ = graph.get_tensor_by_name("x:0")
keep_prob_ = graph.get_tensor_by_name("dropout:0")

# Now, access the op that you want to run.
y_ = graph.get_tensor_by_name("prediction:0")


predictions = sess.run(y_,feed_dict={x_: train_X, keep_prob_: 1})

predictions_val = predictions[:,1]
true_val = train_label[:,1]

fpr, tpr, thresholds  =  roc_curve(true_val, predictions_val)  #计算真正率和假正率
roc_auc = auc(fpr,tpr)                                         #计算auc的值
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
