import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
from data_prep import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_weights1(save_dir,epoch):
	f = np.load(save_dir+"/weights_"+str(epoch)+".npz")
	initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
	
	return initial_weights[0]

def load_weights(epoch):
	f = np.load("weights/weights_"+str(epoch)+".npz")
	initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
	initial_weights = initial_weights[0:2]
	assign_ops = [w.assign(v) for w, v in zip(tf.trainable_variables(), initial_weights)]
	sess.run(tf.global_variables_initializer())
	sess.run(assign_ops)
	print '[INFO] weights loaded from {} epoch'.format(epoch)

def grid_plot(conv1,file_name):
	num_cols = 8
	fig = plt.figure()
	gs = gridspec.GridSpec(num_cols, num_cols, wspace=0.0, hspace=0.0)
	ax = [plt.subplot(gs[i]) for i in range(64)]
	gs.update(hspace=0, wspace=0 )
	
	conv1 = conv1.transpose(3,0,1,2)
	print conv1.shape
	for im,j in zip(conv1, range(len(conv1))):
		im = np.sum(im, axis=2)
		# print im.shape,j
		ax[j].imshow(im, cmap="gray")
		ax[j].axis('off')
	
	plt.savefig('plots/conv_plots/'+file_name )
	plt.cla()

save_dir,epoch = 'weights/', 6
conv1 = load_weights1(save_dir,epoch)
grid_plot(conv1,'conv_weights.png')


x      = tf.placeholder(tf.float32, [None,28,28,1], name='input_node')
conv_1 =  tf.layers.conv2d(inputs = x,filters=64,kernel_size=(3,3),strides=(1, 1),padding='SAME')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epoch = 7
load_weights(epoch)

print '[INFO] Loading the data...'
train = '../data/train.csv'
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
	print '[INFO] conv plot for {}'.format(labels[train_y[img_num]])
	fig = plt.figure()
	plt.axis('off')
	plt.imshow(train_X[img_num].reshape(28,28),cmap='gray_r', interpolation='nearest')
	plt.savefig('plots/conv_plots/img_{}'.format(labels[train_y[img_num]]) + '.png' )
	plt.cla()
	plt.close()
	xs = train_X[img_num].reshape(1,28,28,1)
	outputs = sess.run(conv_1,feed_dict={x:xs})
	conv1   = outputs.transpose(1,2,0,3)
	grid_plot(conv1,'filter_output_{}'.format(labels[train_y[img_num]]) + '.png')
	outputs = outputs[0]
	outputs = np.sum(outputs, axis=2)
	outputs = outputs/np.max(outputs)
	print "Plotting image", img_num
	fig = plt.figure()
	plt.axis('off')
	plt.imshow(outputs, cmap='gray_r', interpolation='nearest')       
	plt.savefig('plots/conv_plots/output_{}'.format(labels[train_y[img_num]]) + '.png')
	plt.cla()

