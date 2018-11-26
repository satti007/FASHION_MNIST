import os
import sys
import random
import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def conv2d(input_layer,filters,ksize,stride,padding,scope):
	with tf.variable_scope(scope) as scope:
		return tf.contrib.layers.convolution2d(inputs=input_layer,num_outputs=filters,
								 kernel_size=ksize,stride=stride,padding=padding,scope=scope)

def max_pool2d(input_layer,ksize,stride):
	return tf.contrib.layers.max_pool2d(inputs=input_layer,kernel_size=ksize,stride=stride,padding='VALID')

def GBP(x,intial_grads):
	with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable]):
		
		# TODO: convlayer: input = 28,28,1  output = 28,28,64
		layer_1 = conv2d(x,64,3,1,'SAME','conv1')
		
		# TODO: poollayer: input = 28,28,64  output = 14,14,64
		pool_1 = max_pool2d(layer_1,2,2)
		
		# TODO: convlayer: input = 14,14,64 output = 14,14,128
		layer_2 = conv2d(pool_1,128,3,1,'SAME','conv2')        
		
		# TODO: poollayer: input = 14,14,128  output = 7,7,128
		pool_2 = max_pool2d(layer_2,2,2)
		
		# TODO: convlayer: input = 7,7,128  output = 7,7,256
		layer_3 = conv2d(pool_2,256,3,1,'SAME','conv3')
		
		# TODO: convlayer: input = 7,7,256  output = 7,7,256
		layer_4 = conv2d(layer_3,256,3,1,'SAME','conv4')
		
		back_prop2    = tf.gradients(layer_4,layer_3,grad_ys=intial_grads)[0]
		back_prop2_th = tf.nn.relu(back_prop2)
		back_prop3    = tf.gradients(layer_3,layer_2,grad_ys = back_prop2_th)[0]
		back_prop3_th = tf.nn.relu(back_prop3)
		back_prop4    = tf.gradients(layer_2,layer_1,grad_ys=back_prop3_th)[0]
		back_prop4_th = tf.nn.relu(back_prop4)
		inp_grads     = tf.nn.relu(tf.gradients(layer_1,x,grad_ys = back_prop4_th)[0])
		
		return inp_grads

# Load the weights in to the graph
def load_weights(epoch):
	f = np.load("weights/weights_"+str(epoch)+".npz")
	initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
	initial_weights = initial_weights[0:8]
	assign_ops = [w.assign(v) for w, v in zip(tf.trainable_variables(), initial_weights)]
	sess.run(tf.global_variables_initializer())
	sess.run(assign_ops)
	print '[INFO] weights loaded from {} epoch'.format(epoch)

x             = tf.placeholder(tf.float32, [None,28,28,1], name='input_node')
initial_grads = tf.placeholder("float", [1,7,7,256])
inp_grads = GBP(x,initial_grads)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epoch = 7
load_weights(epoch)

print '[INFO] Loading the data...'
train = '/home/satti/Documents/Sem8/DL/PA3/data/train.csv'
train_data = pd.read_csv(train).as_matrix()
train_X, train_y = train_data[:,1:785], train_data[:,785]
train_X = train_X/255.0
train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
print '[INFO] Training_data details: ',train_X.shape, train_y.shape

labels = {0: 'Top', 1: 'Trouser', 2: 'Pullover',
		3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
		7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

img_nums = [2426,16,2429,2500,2425,2432,2433,6,2424,2435]
for img_num in img_nums:
	print '[INFO] Guided back_prop for {}'.format(labels[train_y[img_num]])
	os.mkdir('plots/plot_guide_{}'.format(labels[train_y[img_num]]))
	fig = plt.figure()
	plt.axis('off')
	plt.imshow(train_X[img_num].reshape(28,28),cmap='gray_r', interpolation='nearest')
	plt.savefig('plots/plot_guide_{}/img_{}'.format(labels[train_y[img_num]],img_num) + '.png' )
	plt.cla()
	plt.close()
	
	neu_len  = 500
	print "index, neuron number" 
	good = 0
	for n in range(neu_len):
		neu_num   = np.random.randint(12000)
		initial_t = ([0.0]*(7*7*256))
		initial_t[neu_num - 1] = 1.0
		n = n + 1
		print(n, neu_num)
		initial_t = np.asarray(initial_t).reshape(1,7,7,256)
		xs = train_X[img_num].reshape(1,28,28,1)
		grads = sess.run(inp_grads,feed_dict={x:xs,initial_grads:initial_t})
		grads = grads[0]
		grads = np.sum(grads, axis=2)
		if(np.count_nonzero(grads)!=0):
			grads = grads/np.max(grads)
			print "Plotting image", n
			plt.axis('off')
			plt.imshow(grads, cmap='gray_r', interpolation='nearest')       
			plt.savefig('plots/plot_guide_{}/guided_{}'.format(labels[train_y[img_num]],neu_num) + '.png')
			plt.cla()
			good += 1
		else:
			print 'Not useful'
	
	print good