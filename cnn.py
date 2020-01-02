import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.set_random_seed(777)

pos_nums = open('rt-polarity.pos','r',encoding='utf-8').readlines()
pos_labels = [[0,1] for _ in range(len(pos_nums))]
neg_nums = open('rt-polarity.neg','r',encoding='utf-8').readlines()
neg_labels = [[1,0] for _ in range(len(neg_nums))]
data = pos_nums+neg_nums
y_data = np.array(pos_labels + neg_labels)

# 句子最大长度
max_length = max([len(i.split(' ')) for i in pos_nums])
print('句子最大长度',max_length)

vocab = learn.preprocessing.VocabularyProcessor(max_document_length=max_length,)
x_data = np.array(list(vocab.fit_transform(data)))

# 洗牌
np.random.seed(0)
order = np.random.permutation(len(y_data))
x_data = x_data[order]
y_data = y_data[order]

n = int(len(y_data) * 0.7)
trainx,testx = np.split(x_data,[n])
trainy,testy = np.split(y_data,[n])

# 占位符
X = tf.placeholder(dtype=tf.int32,shape=[None,max_length])
Y = tf.placeholder(dtype=tf.float32,shape=[None,2])
keep_prob = tf.placeholder(tf.float32)
# 词嵌入
w = tf.Variable(tf.random_normal([len(vocab.vocabulary_),8],stddev=0.01))
x_embed = tf.nn.embedding_lookup(w,X)

x_cnn = tf.reshape(x_embed,[-1,61,8,1])

# 卷积1
w1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
L1 = tf.nn.conv2d(x_cnn,w1,strides=[1,1,1,1],padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L1 = tf.nn.dropout(L1,keep_prob=keep_prob)

# 卷积2
w2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
L2 = tf.nn.conv2d(L1,w2,strides=[1,1,1,1],padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L2 = tf.nn.dropout(L2,keep_prob=keep_prob)

dim = L2.get_shape()[1].value*L2.get_shape()[2].value*L2.get_shape()[3].value
L2_flat = tf.reshape(L2,[-1,dim])

# 全连接1
w3 = tf.Variable(tf.random_normal([dim,2],stddev=0.01))
b = tf.Variable(tf.random_normal([2],stddev=0.01))
logits = tf.matmul(L2_flat,w3) + b

# 准确率
pre = tf.argmax(logits,axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(pre,tf.argmax(Y,axis=1)),dtype=tf.float32))
# 代价
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
tf.summary.scalar('cost',cost)
summary = tf.summary.merge_all()
# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

train_epochs = 15
batch_size = 100
global_step = 0
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/text1',session.graph)
    for i in range(train_epochs):
        totals = int(len(trainy)/batch_size)
        for j in range(totals):
            x_bat = trainx[batch_size*j:batch_size*(j+1)]
            y_bat = trainy[batch_size*j:batch_size*(j+1)]
            cost_,acc_,_,s = session.run([cost,acc,optimizer,summary],feed_dict={X:x_bat,Y:y_bat,keep_prob:0.7})
            print(i,j,'cost=',cost_,'acc=',acc_)
            writer.add_summary(s,global_step=global_step)
            global_step += 1

    acc_= session.run(acc, feed_dict={X: testx, Y: testy, keep_prob: 1.0})
    print('准确率:',acc_)
















