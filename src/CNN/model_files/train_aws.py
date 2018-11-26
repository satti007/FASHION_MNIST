import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
from data_prep import *

tf.set_random_seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lr,batch_size,init,save_dir = get_arguments() # Parsing the argumnets
train,valid,test            = '../data/train.csv','../data/val.csv','../data/test.csv'
train_X,train_y,valid_X,valid_y, test_X   = load_data(train,valid,test) # Loading the data

global w,drop_flag,BN_flag

# weight intilization
if init:
	w=1.0
	mod="FAN_AVG"
else:
	w=2.0
	mod="FAN_IN"

drop_flag, BN_flag = False, True

# Given input --> returns activ_fun[conv2d(input,W,b)]
def conv2d(input_layer,filters,ksize,stride,padding,scope):
	with tf.variable_scope(scope) as scope:
		return tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=filters,
				weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=w,mode=mod),
				kernel_size=ksize,stride=stride,padding=padding,scope=scope)

# Given input --> returns maxpooled output
def max_pool2d(input_layer,ksize,stride):
	return tf.contrib.layers.max_pool2d(inputs=input_layer,kernel_size=ksize,stride=stride,padding='VALID')

# Given input --> returns activ_fun[input*W+b]
def FC(input_layer,neurons,scope):
	with tf.variable_scope(scope) as scope:
		return tf.contrib.layers.fully_connected(inputs=input_layer,num_outputs=neurons,
				weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=w,mode=mod),scope=scope)

# Given input --> returns input*W+b
def softmax(input_layer,neurons,scope,isTraining):
	with tf.variable_scope(scope) as scope:
		if BN_flag:
			return tf.contrib.layers.fully_connected(inputs=input_layer,num_outputs=neurons,activation_fn=None,
			scope=scope,weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=w,mode=mod),	
			normalizer_fn=tf.contrib.layers.batch_norm,normalizer_params={'is_training':isTraining,'updates_collections':None,'scale':True})
		return tf.contrib.layers.fully_connected(inputs=input_layer,num_outputs=neurons,activation_fn=None,
		weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=w,mode=mod),scope=scope)

# Neural network -- model
def model(x,isTraining,keep_prob1):
	with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable]):
		
		# TODO: convlayer: input = 28,28,1  output = 28,28,64
		layer_1 = conv2d(x,64,3,1,'SAME','conv1')
		
		# TODO: poollayer: input = 28,28,64  output = 14,14,64
		pool_1  = max_pool2d(layer_1,2,2)
		
		# TODO: convlayer: input = 14,14,64 output = 14,14,128
		layer_2 = conv2d(pool_1,128,3,1,'SAME','conv2')        
		
		# TODO: poollayer: input = 14,14,128  output = 7,7,128
		pool_2  = max_pool2d(layer_2,2,2)
		
		# TODO: convlayer: input = 7,7,128  output = 7,7,256
		layer_3 = conv2d(pool_2,256,3,1,'SAME','conv3')
		
		# TODO: convlayer: input = 7,7,256  output = 7,7,256
		layer_4 = conv2d(layer_3,256,3,1,'SAME','conv4')
		
		# TODO: poollayer: input = 7,7,256  output = 3,3,256
		pool_3  = max_pool2d(layer_4,2,2)
		
		# TODO: flatten layer: input = 3,3,256  output = 2304
		layer_f = tf.contrib.layers.flatten(pool_3)
		
		# TODO: FC layer: input = 2304 output = 1024
		layer_5  = FC(layer_f,1024,'fc5')
		if drop_flag:
			layer_5 = tf.nn.dropout(layer_5,keep_prob=keep_prob1) # dropout layer
		
		# TODO: FC layer: input = 1024  output = 1024
		layer_6  = FC(layer_5,1024,'fc6') 
		if drop_flag:
			layer_6 = tf.nn.dropout(layer_6,keep_prob=keep_prob1) # dropout layer
		
		# TODO: FC layer: input = 1024  output = 10
		logits   = softmax(layer_6,10,'fc7',isTraining) # Batch Normalization
		
		# TODO: softmax layer: input = 10  output = 10
		y        = tf.nn.softmax(logits,name='output_node')
		
		return logits,y

# Save the weights in numpy arrays 
def save_weights(epoch):
	Wts = [p.eval(session=sess) for p in tf.trainable_variables()]
	np.savez(save_dir+"/weights_"+str(epoch)+".npz", *Wts)
	print '[INFO] weights saved at {} epoch'.format(epoch)

# Load the weights in to the graph
def load_weights(epoch):
	f = np.load(save_dir+"/weights_"+str(epoch)+".npz")
	initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
	assign_ops = [w.assign(v) for w, v in zip(tf.trainable_variables(), initial_weights)]
	sess.run(tf.global_variables_initializer())
	sess.run(assign_ops)
	print '[INFO] weights loaded from {} epoch'.format(epoch)

# Caluclate accuracy given true_one_hot and predicted_one_hot
def accuracy(y,y_):
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	return sess.run(acc)

# Define the placeholders, loss, training details
x  = tf.placeholder(tf.float32, [None,28,28,1], name='input_node')
y_ = tf.placeholder("float",shape=[None,10])
keep_prob1    = tf.placeholder(tf.float32)
isTraining    = tf.placeholder(tf.bool,name='isTraining')
logits,y      = model(x,isTraining,keep_prob1)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
train_step    = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
sess          = tf.Session()
sess.run(tf.global_variables_initializer())

# Testing by loading weights at given epoch
def dotest(epoch,test_X):
	load_weights(epoch)
	file = open('sub_{}.csv'.format(epoch),'w')
	file.write('id,label\n')
	steps = test_X.shape[0]/10
	for s in range(steps):
		xs,ids = test_X[10*s:10*s+10],list(np.arange(10*s,10*s+10))
		pred   = sess.run(y,feed_dict={x:xs,isTraining:False,keep_prob1:1})
		labels = sess.run(tf.argmax(pred, 1))
		for i,l in zip(ids,labels):
			file.write("{},{}\n".format(i,l))
		print'[INFO] Step:{}, labels:{}'.format(s,labels)
	file.close()

# Get TOTAL traning & validation data loss, accuracy after every epoch
def show_loss_acc(isTrain,train_X,train_y,valid_X,valid_y,epoch,step):
	Vbatch_size,loss,acc = 1000,0,0
	if isTrain:
		S    = train_X.shape[0]/Vbatch_size
		data = 'Total_train'
	else:
		S    = valid_X.shape[0]/Vbatch_size
		data = 'Total_valid'
	for i in range(1, S+1):
		xs,ys = get_batch_data(Vbatch_size,train_X,train_y,valid_X,valid_y,i,isTrain)
		loss += sess.run(cross_entropy,feed_dict = {x:xs,y_:ys,isTraining:True,keep_prob1:1})
		pred  = sess.run(y,feed_dict = {x:xs,isTraining:isTrain,keep_prob1:1})
		acc  += accuracy(pred.reshape(Vbatch_size,10),ys)
	print     "[INFO] Epoch:{}, Step: {}, {}_epoch_loss: {}, {}_epoch_acc:{}".format(epoch,step,data,round(loss/S,3),data,100*round(acc/S,3))
	log.write("[INFO] Epoch:{}, Step: {}, {}_epoch_loss: {}, {}_epoch_acc:{}\n".format(epoch,step,data,round(loss/S,3),data,100*round(acc/S,3)))
	
	return round(loss/S,3),100*round(acc/S,3)

max_epochs,show,prob,EA_flag = 20,100,0.6,False
log = open('train.log','a',0) # logfile
print     'Param details:{},{},{}'.format(lr,batch_size,init)
log.write('Param details:{},{},{}\n'.format(lr,batch_size,init))
def train(train_X,train_y,valid_X,valid_y,test_X,EA_flag,load=None,state=None):
	if(load): # load the weights 
		load_weights(state)
	epoch,start = 1,1
	best_val_acc,best_epoch = 0,1
	'''
	Early stopping: 
	ESA -- val acc after every epoch, ESC -- patience count, ESP -- patience
	when(ESC == ESP) : stop traning
	'''
	ESA,ESC,ESP = 0,0,5
	while (epoch <= max_epochs):
		start_time = time.time()
		indicies = random.sample(range(0,train_X.shape[0]),train_X.shape[0]) # shuffle the train data
		train_X,train_y = train_X[indicies], train_y[indicies]
		for step in range(1, train_X.shape[0]/batch_size + 1): # train for (train_X.shape[0]/batch_size) steps
			xs,ys = get_batch_data(batch_size,train_X,train_y,valid_X,valid_y,step,True) # get batch_data
			sess.run(train_step, feed_dict = {x:xs,y_:ys,isTraining:True,keep_prob1:prob}) # do training
			if step % show == 0 or step == (train_X.shape[0]/batch_size): # after every 100 steps show batch loss, acc
				data = 'batch_train'
				loss = sess.run(cross_entropy,feed_dict = {x:xs,y_:ys,isTraining:True,keep_prob1:1})
				pred = sess.run(y,feed_dict = {x:xs,isTraining:True,keep_prob1:1})
				acc  = accuracy(pred.reshape(batch_size,10),ys)
				print     "[INFO] Epoch:{}, Step: {}, {}_loss: {}, {}_acc:{}".format(epoch,step,data,round(loss,3),data,100*round(acc,3))
				log.write("[INFO] Epoch:{}, Step: {}, {}_loss: {}, {}_acc:{}\n".format(epoch,step,data,round(loss,3),data,100*round(acc,3)))
			if step == (train_X.shape[0]/batch_size) or step == 1: # Get TOTAL traning & val data loss, acc after every epoch
				if start == 0 and step == 1:
					continue
				start = 0
				L,A  = show_loss_acc(True,train_X,train_y,valid_X,valid_y,epoch,step)
				L,A  = show_loss_acc(False,train_X,train_y,valid_X,valid_y,epoch,step)
				if A >= best_val_acc:
					best_val_acc,best_epoch = A,epoch
				if EA_flag: # Early stopping
					if A >= ESA:
						ESC = 0
					elif ESC == ESP:
						dotest(best_epoch,test_X)
						print     '[INFO] Ran out of patience at epoch: {}, best_epoch: {}'.format(epoch,best_epoch)
						log.write('[INFO] Ran out of patience at epoch: {}, best_epoch: {}'.format(epoch,best_epoch))
						sys.exit()
					else:
						ESC += 1
						print     '[INFO] Patience left: {} at epoch: {}, best_epoch: {}'.format(ESP-ESC,epoch,best_epoc)
						log.write('[INFO] Patience left: {} at epoch: {}, best_epoch: {}'.format(ESP-ESC,epoch,best_epoc))
		ESA = A
		save_weights(epoch)
		print     'Time taken to complete epoch: {}  is: {}'.format(epoch,(time.time() - start_time))
		log.write('Time taken to complete epoch: {}  is: {}\n'.format(epoch,(time.time() - start_time)))
		epoch += 1
	dotest(best_epoch,test_X)

# train(train_X,train_y,valid_X,valid_y,test_X,EA_flag)
dotest(6,test_X)