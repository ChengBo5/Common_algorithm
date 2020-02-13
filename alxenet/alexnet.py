import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.utils import shuffle

IMG_SIZE = 224

def get_files(file_dir):   # file_dir: 文件夹路径   return: 乱序后的图片和标签
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    # 载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        # name的形式为['dog', '9981', 'jpg']
        # os.listdir将名字转换为列表表达
        if name[0] == 'cat':
            cats.append(file_dir + file)
            # 注意文件路径和名字之间要加分隔符，不然后面查找图片会提示找不到图片
            # 或者在后面传路径的时候末尾加两//  'D:/Python/neural network/Cats_vs_Dogs/data/train//'
            label_cats.append((1,0))
        else:
            dogs.append(file_dir + file)
            label_dogs.append((0,1))
        # 猫为(1,0)，狗为(0,1)
    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))
    # np.hstack()方法将猫和狗图片和标签整合到一起,标签也整合到一起
    image_list = cats + dogs
    label_list = label_cats+label_dogs

    # 打乱文件顺序
    temp = np.array([image_list, label_list])
    # 这里的数组出来的是2行10列，第一行是image_list的数据，第二行是label_list的数据
    temp = temp.transpose()  # 转置
    # 将其转换为10行2列，第一列是image_list的数据，第二列是label_list的数据
    np.random.shuffle(temp)
    # 对应的打乱顺序
    image_list = list(temp[:, 0])  # 取所有行的第0列数据
    label_list = list(temp[:, 1])  # 取所有行的第1列数据，并转换为int
    return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将原来的python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)  # 强制类型转换
    label = tf.cast(label, tf.int32)

    # 生成队列。我们使用slice_input_producer()来建立一个队列，将image和label放入一个list中当做参数传给该函数
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])
    # 按队列读数据和标签
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 要按照图片格式进行解码。本例程中训练数据是jpg格式的，所以使用decode_jpeg()解码器，
    # 如果是其他格式，就要用其他geshi具体可以从官方API中查询。
    # 注意decode出来的数据类型是uint8，之后模型卷积层里面conv2d()要求输入数据为float32类型

    # 统一图片大小
    # 通过裁剪统一,包括裁剪和扩充
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 我的方法，通过缩小图片，采用NEAREST_NEIGHBOR插值方法
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                   align_corners=False)
    image = tf.cast(image, tf.float32)/255.0
    # 因为没有标准化，所以需要转换类型
    # image = tf.image.per_image_standardization(image)   # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,  # 线程
                                              capacity=capacity)
    # image_batch是一个4D的tensor，[batch, width, height, channels]，
    # label_batch是一个2D的tensor，[batch]。
    # 这行多余？
    return image_batch, label_batch

class alexnet(object):
    def __init__(self):
        self.weights = {
            'w1': self.creat_weights([11,11,3,96]),
            'w2': self.creat_weights([5,5,96,256]),
            'w3': self.creat_weights([3,3,256,384]),
            'w4': self.creat_weights([3,3,384,384]),
            'w5': self.creat_weights([3,3,384,256]),
            'w6': self.creat_weights([6*6*256,1024]),
            'w7': self.creat_weights([1024,1024]),
            'w8': self.creat_weights([1024,2]),
        }

        self.biases = {
            'b1': self.creat_bias([96]),
            'b2': self.creat_bias([256]),
            'b3': self.creat_bias([384]),
            'b4': self.creat_bias([384]),
            'b5': self.creat_bias([256]),
            'b6': self.creat_bias([1024]),
            'b7': self.creat_bias([1024]),
            'b8': self.creat_bias([2]),
        }
        self.STEP = 1000
        self.BATCH = 50

    def creat_weights(self,shape):
        return tf.Variable(tf.random_normal(shape = shape,mean = 0,stddev=0.01))

    def creat_bias(self,shape):
        return tf.Variable(tf.constant(0.01,shape= shape))

    def conv2d(self,input,filter,bias,strides = None):
        if not strides:
            strides = [1,1,1,1]
        conv = tf.nn.conv2d(input=input,filter=filter,strides=strides,padding='SAME')
        conv = tf.nn.relu(tf.nn.bias_add(conv,bias))
        return conv

    def lrn(self,input):
        return tf.nn.lrn(input=input,alpha=1e-4,beta=75,depth_radius=2,bias=2.0)

    def pooling(self,input):
        return tf.nn.max_pool(input,ksize=[1,3,3,1],strides=[1,2,2,1],padding = 'VALID')

    def nn(self,input):
        with tf.name_scope('conv1') as scope:
            conv1 = self.conv2d(input,self.weights['w1'],self.biases['b1'],[1,4,4,1])
            conv1 = self.lrn(conv1)
            conv1 = self.pooling(conv1)

        with tf.name_scope('conv2') as scope:
            conv2 = self.conv2d(conv1,self.weights['w2'],self.biases['b2'])
            conv2 = self.lrn(conv2)
            conv2 = self.pooling(conv2)

        with tf.name_scope('conv3') as scope:
            conv3 = self.conv2d(conv2,self.weights['w3'],self.biases['b3'],)

        with tf.name_scope('conv4') as scope:
            conv4 = self.conv2d(conv3,self.weights['w4'],self.biases['b4'])

        with tf.name_scope('conv5') as scope:
            conv5 = self.conv2d(conv4,self.weights['w5'],self.biases['b5'],[1,4,4,1])
            conv5 = self.lrn(conv5)
            conv5 = self.pooling(conv5)

        with tf.name_scope('fc6') as scope:
            reshape = tf.reshape(conv5,[-1,conv5.get_shape()[1].value*
                                 conv5.get_shape()[2].value*
                                 conv5.get_shape()[2].value])
            fc6 = tf.nn.relu(conv5,tf.matmul(reshape,self.weights['w6'])+self.bisaes['b6'])
            fc6 = tf.nn.dropout(fc6,8)

        with tf.name_scope('fc7') as scope:
            fc7 = tf.nn.relu(tf.matmul(fc6,self.weights['w7'])+self.bisaes['b7'])
            fc7 = tf.nn.dropout(fc7,8)

        with tf.name_scope('fc8') as scope:
            fc8 = tf.nn.relu(tf.matmul(fc7,self.weights['w8'])+self.bisaes['b8'])

        return fc8

    def train(self,img_path = None,isSave = False , isTrain = True):
        x = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
        y_true = tf.placeholder(dtype=tf.float32,shape=[None,2])
        y_pred = self.nn(x)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
        op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true,1),tf.argmax(y_pred,1))))
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            if isTrain:
                for i in range(self.STEP):
                    image_list, label_list = get_files('kaggle/train/')
                    train_batch, train_label_batch = get_batch(image_list, label_list, 224, 224, 50, 10)
                    # x_= images[i*self.BATCH:(i+1)*self.BATCH]
                    # y_= labels[i*self.BATCH:(i+1)*self.BATCH]

                    _,lossval= sess.run({op,loss},feed_dict={x:train_batch,y_true:train_label_batch})
                    print('训练第%d步，损失为：%f' % (i,lossval))
                    if isSave:
                        if i%100 ==0:
                            saver.save(sess,'model/alexnet_wights')

            else:
                saver.restore(sess,'model/alexnet_wights')
                x_test= np.reshape(cv2.resize(cv2.imread(img_path),(224,224))
                    [-1,224,224,3].astype('float32'))
                y_test= np.zeros([1,2])
                predict = tf.argmax(sess.run(y_pred, feed_dict={x:x_test,y_true:y_test}))
                return predict

net =alexnet()
net.train(isSave=True,)
print("over")
