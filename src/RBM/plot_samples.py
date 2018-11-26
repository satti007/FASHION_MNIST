import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from GSampling import *

np.random.seed(1234)

# sampling
def do_sampling(weights,v_biases,h_biases,k_step,v_dim,h_dim,xs):
	V_d = xs.reshape(v_dim,1)
	for j in range(0,k_step):
		H_t = gibbs_sample_RBM(weights,h_biases,v_biases,False,True,V_d)
		V_T = gibbs_sample_RBM(weights,h_biases,v_biases,True,False,None,H_t)
		V_d = np.copy(V_T)
	
	return V_T.reshape(28,28)

# load the weights at a given state
def load_weights(save_dir,state):
	weights = np.load(save_dir+'/weights_{}.npy'.format(state))
	biases  = list(np.load(save_dir+'/biases_{}.npy'.format(state)))
	v_biases,h_biases = biases[0],biases[1]
	# print '[INFO] Model restored at {} epoch'.format(state)
	
	return weights,v_biases,h_biases

# plot samples after each epoch in a 8x8 grid
def grid_plot(sample,file_name):
	num_cols = 8
	fig = plt.figure()
	gs = gridspec.GridSpec(num_cols, num_cols, wspace=0.0, hspace=0.0)
	ax = [plt.subplot(gs[i]) for i in range(64)]
	gs.update(hspace=0, wspace=0 )
	
	for im,j in zip(sample, range(len(sample))):
		ax[j].imshow(im, cmap="gray_r")
		ax[j].axis('off')
	
	plt.savefig('plots/sample_plots/'+file_name )
	plt.cla()


save_dir = 'weights/weights_n_700'
EPOCHS = 64

print '[INFO] Loading the data...'
train = '../data/sample.csv'
train_data = pd.read_csv(train).as_matrix()
train_X, train_y = train_data[:,1:785], train_data[:,785]
data_X = train_X/127
data_X[data_X==2] = 1
train_X = train_X/255.0
print '[INFO] Training_data details: ',train_X.shape, train_y.shape

labels = {0: 'Top', 1: 'Trouser', 2: 'Pullover',
		3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
		7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

img_nums = [2426,16,2429,2500,2425,2432,2433,6,2424,2435]

k_step,v_dim,h_dim = 1,784,700
for img_num in img_nums:
	print '[INFO] sample plots for {}'.format(labels[train_y[img_nums.index(img_num)]])
	fig = plt.figure()
	plt.axis('off')
	plt.imshow(train_X[img_nums.index(img_num)].reshape(28,28),cmap='gray_r', interpolation='nearest')
	plt.savefig('plots/sample_plots/img_{}'.format(labels[train_y[img_nums.index(img_num)]]) + '.png' )
	plt.cla()
	plt.close()
	
	xs     = data_X[img_nums.index(img_num)]
	sample = np.zeros([EPOCHS,28,28])
	for epoch in range(1,EPOCHS+1):
		weights,v_biases,h_biases = load_weights(save_dir,epoch)
		sample[epoch-1]  = do_sampling(weights,v_biases,h_biases,k_step,v_dim,h_dim,xs)
	
	grid_plot(sample,'samples_{}'.format(labels[train_y[img_nums.index(img_num)]]) + '.png')
