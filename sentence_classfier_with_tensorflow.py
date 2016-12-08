# -*- coding: utf-8 -*-
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import re
import warnings
import sys
import time
import numpy as np
import os
import time
import datetime
warnings.filterwarnings("ignore") 
import tensorflow as tf
from tensorflow.contrib import learn
"""
使用tensorflow构建CNN模型进行文本分类
"""
vector_size = 100;
sentence_length = 100;
hidden_layer_input_size = 100
filter_hs=[3,4,5]
num_filters = 128
img_h = sentence_length
img_w = vector_size
filter_w = img_w
batch_size=100
word_idx_map_szie = 75924#18766#75924

#一些数据预处理的方法====================================== 
def get_idx_from_sent(sent, word_idx_map, max_l):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    #一个训练的一个输入 形式为[0,0,0,0,x11,x12,,,,0,0,0] 向量长度为max_l+2*filter_h-2
    return x 
def generate_batch(minibatch_index):
    minibatch_data = revs[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
    batchs = np.ndarray(shape=(batch_size, sentence_length), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 2), dtype=np.int32)
    
    for i in range(batch_size):
        sentece = minibatch_data[i]["text"]
        lable =  minibatch_data[i]["y"]
        if lable==1:
            labels[i] = [0,1]#
        else:
            labels[i] = [1,0]#
        batch = get_idx_from_sent(sentece, word_idx_map, sentence_length)
        batchs[i] = batch
    return batchs, labels
def get_test_batch(cv=1):
    test = []
    for rev in revs:
        if rev["split"]==cv:
            test.append(rev)        
    minibatch_data = test
    test_szie =len(minibatch_data)
    batchs = np.ndarray(shape=(test_szie, sentence_length), dtype=np.int32)
    labels = np.ndarray(shape=(test_szie, 2), dtype=np.int32)  
    for i in range(test_szie):
        sentece = minibatch_data[i]["text"]
        lable =  minibatch_data[i]["y"]
        if lable==1:
            labels[i] = [0,1]
        else:
            labels[i] = [1,0]
        batch = get_idx_from_sent(sentece, word_idx_map, sentence_length)
        batchs[i] = batch
    return batchs, labels
print "loading data...",
x = cPickle.load(open("mr.p","rb"))
# 读取出预处理后的数据 revs {"y":label,"text":"word1 word2 ..."}
#                          word_idx_map["word"]==>index
#                        vocab["word"]==>frequency
revs, _, _, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4] 
print "data loaded!"
revs = np.random.permutation(revs) #原始的sample正负样本是分别聚在一起的，这里随机打散
n_batches = len(revs)/batch_size#
n_train_batches = int(np.round(n_batches*0.9))

#开始定义模型============================================
sess = tf.InteractiveSession()
 
#占位符 真实的输入输出
x_in = tf.placeholder(tf.int32, shape=[None,sentence_length],name="input_x")
y_in = tf.placeholder("float", [None,2],name="input_y")#2分类问题

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#产生常量矩阵
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积图层 第一个卷积
# strides：每跨多少步抽取信息，strides[1, x_movement,y_movement, 1]， [0]和strides[3]必须为1
# padding：边距处理，“SAME”表示输出图层和输入图层大小保持不变，设置为“ VALID ”时表示舍弃多余边距(丢失信息)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
# 定义pooling图层
# pooling：解决跨步大时可能丢失一些信息的问题,max-pooling就是在前图层上依次不重合采样2*2的窗口最大值
def max_pool_2x2(x, filter_h):
    return tf.nn.max_pool(x, ksize=[1, img_h-filter_h+1, 1, 1],strides=[1, 1, 1, 1], padding='VALID')


# Embedding layer===============================
#要学习的词向量矩阵
embeddings = tf.Variable(tf.random_uniform([word_idx_map_szie, vector_size], -1.0, 1.0)) 
#输入reshape
x_image_tmp = tf.nn.embedding_lookup(embeddings, x_in)
#输入size: sentence_length*vector_size
#x_image = tf.reshape(x_image_tmp, [-1,sentence_length,vector_size,1])======>>>>>
#将[None, sequence_length, embedding_size]转为[None, sequence_length, embedding_size, 1]
x_image = tf.expand_dims(x_image_tmp, -1)#单通道

#定义卷积层===================================
W_conv = []
b_conv = []
for filter_h in filter_hs:
    #卷积的patch大小：vector_size*filter_h, 通道数量：1, 卷积数量：hidden_layer_input_size
    filter_shape = [filter_h, vector_size, 1, num_filters]
    W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
    #W_conv1 = weight_variable([filter_h, vector_size, 1, hidden_layer_input_size])
    W_conv.append(W_conv1)
    #b_conv1 = bias_variable([hidden_layer_input_size])
    b_conv.append(b_conv1)
#进行卷积操作
h_conv = []
for W_conv1,b_conv1 in zip(W_conv,b_conv):
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #输出szie: (sentence_length-filter_h+1,1)
    h_conv.append(h_conv1)

#pool层========================================
h_pool_output = []
for h_conv1,filter_h in zip(h_conv,filter_hs):
    h_pool1 = max_pool_2x2(h_conv1, filter_h) #输出szie:1
    h_pool_output.append(h_pool1)
    

#全连接层=========================================
l2_reg_lambda=0.001
#输入reshape
num_filters_total = num_filters * len(filter_hs)
h_pool = tf.concat(3, h_pool_output)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
#h_pool_flat = tf.reshape(h_pool, [-1, hidden_layer_input_size*len(filter_hs)])

keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h_pool_flat, keep_prob)
#W = tf.Variable(tf.random_uniform([num_filters_total, 2], -1.0, 1.0)) 
W = tf.Variable(tf.truncated_normal([num_filters_total, 2], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
l2_loss = tf.nn.l2_loss(W)
l2_loss += tf.nn.l2_loss(b)

#scores=tf.nn.softmax(tf.matmul(h_drop, W) + b) #a softmax function to convert raw scores into normalized probabilities
scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores") # wx+b

predictions = tf.argmax(scores, 1, name="predictions")
losses = tf.nn.softmax_cross_entropy_with_logits(scores, y_in)
loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

correct_prediction = tf.equal(tf.argmax(scores,1), tf.argmax(y_in,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#train-modle========================================
num_steps = 4000
#L2正则化参数
beta=0.001
global_step = tf.Variable(0) 

learning_rate = tf.train.exponential_decay(1e-4, global_step, num_steps, 0.99, staircase=True)#学习率递减

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss,  global_step=global_step)

#summaries,====================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
loss_summary = tf.scalar_summary("loss", loss)
acc_summary = tf.scalar_summary("accuracy", accuracy)
train_summary_op = tf.merge_summary([loss_summary, acc_summary])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.all_variables())


sess.run(tf.initialize_all_variables())

ckpt = tf.train.get_checkpoint_state("/home/conv_net_sentence/runs/1481160097/checkpoints")  
if ckpt and ckpt.model_checkpoint_path:  
    saver.restore(sess, ckpt.model_checkpoint_path)  

batch_x_test, batch_y_test = get_test_batch()
current_step = tf.train.global_step(sess, global_step)
print "current_step:", current_step
if num_steps>int(current_step/135):
    num_steps = num_steps - int(current_step/135)
    print "continute step:",num_steps 
else:
    num_steps = 0
for i in range(num_steps):
    for minibatch_index in np.random.permutation(range(n_train_batches)): #随机打散 每次输入的样本的顺序都不一样
        batch_x, batch_y = generate_batch(minibatch_index)
        #train_step.run(feed_dict={x_in: batch_x, y_in: batch_y, keep_prob: 0.5})
        feed_dict={x_in: batch_x, y_in: batch_y, keep_prob: 0.5}
        _, step, summaries = sess.run([train_step, global_step, train_summary_op],feed_dict)
        train_summary_writer.add_summary(summaries, step)
    #print batch_x[0]
    #print batch_y[0]
    train_accuracy = accuracy.eval(feed_dict={x_in:batch_x_test, y_in: batch_y_test, keep_prob: 1.0})
    current_step = tf.train.global_step(sess, global_step)
    print "step %d, training accuracy %g"%(current_step, train_accuracy)
    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
    print("Saved model checkpoint to {}\n".format(path)) 

"""
#下次调用
saver = tf.train.Saver()
with tf.Session() as sess: 
    saver.restore(sess, '/root/alexnet.tfmodel')
    sess.run(....)
"""   
 
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm  
final_embeddings = normalized_embeddings.eval() 
filename = "CNN_result_final_embeddings"
cPickle.dump(final_embeddings, open(filename, "wb"))
"""
#可视化  
from matplotlib import pylab
from sklearn.manifold import TSNE

num_points = 400
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])

def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()


words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)   
"""   
    
    
    
    
    
    
    
    
    
    
    
    
    
    





