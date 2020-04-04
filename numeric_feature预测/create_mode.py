import tensorflow as tf
import numpy as np
import pandas as pd


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
    train_label = pd.get_dummies(label, drop_first=False)

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

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,column_x] , name='x')
y = tf.placeholder(tf.float32,[None,class_num] , name='y')
keep_prob=tf.placeholder(tf.float32 , name='dropout')

#创建一个简单的神经网络
W1 = tf.Variable(tf.truncated_normal([column_x,500],stddev=0.1))
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([500,500],stddev=0.1))
b2 = tf.Variable(tf.zeros([500])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([500,100],stddev=0.1))
b3 = tf.Variable(tf.zeros([100])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob)

W4 = tf.Variable(tf.truncated_normal([100,2],stddev=0.1))
b4 = tf.Variable(tf.zeros([2])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4)+b4 , name='prediction')

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        sess.run(train_step,feed_dict={x:train_X,y:train_label,keep_prob:0.7})
        if epoch % 10 == 0 :
            loss_val,test_acc = sess.run([loss,accuracy],feed_dict={x:train_X,y:train_label,keep_prob:1.0})
            print("Iter " + str(epoch) + ",loss_val " + str(loss_val) + ",Testing Accuracy " + str(test_acc))

    saver = tf.train.Saver(max_to_keep=4)
    tf.train.Saver().save(sess, save_path="models/model",write_meta_graph=True)
























