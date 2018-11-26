# To run this file
'''
python -W ignore PCA.py --red_dim 100\
		--train ../../data/train.csv\
		--valid ../../data/valid.csv\
		--test ../../data/test.csv\
		--save_dir ../../data
'''

import os
import random
import argparse
import numpy as np
import pandas as pd
np.random.seed(1234)


def get_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('--train', type=str)
	ap.add_argument('--valid', type=str)
	ap.add_argument('--test' , type=str)
	ap.add_argument('--save_dir', type=str)
	ap.add_argument('--red_dim' , type=int,default=-1)
	
	print '[INFO] Parsing the Arguments...'
	args = vars(ap.parse_args())
	
	train = args['train']
	valid = args['valid']
	test  = args['test']
	red_dim  = args['red_dim']
	save_dir = args['save_dir']
	print 'train: {}\nvalid: {}\ntest: {}'.format(train,valid,test)
	print 'save_dir: {}\nreduce_dim to: {}'.format(save_dir,red_dim)
	
	return train,valid,test,save_dir,red_dim

# change range to [0,1] 
def do_normalizing(train_X,valid_X,test_X):
	train_X = train_X/float(np.max(train_X))
	valid_X = valid_X/float(np.max(valid_X))
	test_X  = test_X/float(np.max(test_X))
	
	return train_X,valid_X,test_X

def load_data(train,valid,test):
	print '[INFO] Loading the data...'
	train_data = pd.read_csv(train).as_matrix()
	valid_data = pd.read_csv(valid).as_matrix()
	test_data = pd.read_csv(test).as_matrix()
	in_dim     = train_data.shape[1]-2 # assuming there is index at first col & label at the last col
	train_X, train_y = train_data[:,1:in_dim+1], train_data[:,in_dim+1]
	valid_X, valid_y = valid_data[:,1:in_dim+1], valid_data[:,in_dim+1]
	test_X = test_data[:,1:in_dim+1]
	print '[INFO] Train_data details: ',train_X.shape, train_y.shape
	print '[INFO] Valid_data details: ',valid_X.shape, valid_y.shape
	print '[INFO]  Test_data details: ',test_X.shape
	print '[INFO] Reading the data Done!'
	
	return train_X,train_y,valid_X,valid_y,test_X

def do_PCA(train_X,train_y,valid_X,valid_y,test_X,save_dir,red_dim):
	print '[INFO] PCA started...'
	sigma   = np.cov(train_X,rowvar=False)
	U,S,V   = np.linalg.svd(sigma,compute_uv=1)
	var_ret = np.cumsum(S)/np.sum(S)
	if red_dim == -1:
		K = np.argwhere((var_ret > 0.90)&(var_ret < 0.99))
		K = [K[0][0]+1,K[-1][0]+1]
	else:
		K = [red_dim]
	
	for k in K:
		Z = U[:, 0:k]
		os.system('mkdir -p {}/dim_{}'.format(save_dir,k))
		pd.DataFrame(np.hstack((np.matmul(train_X,Z),train_y.reshape(train_y.shape[0],1)))).to_csv(save_dir+'/dim_{}/train.csv'.format(k))
		pd.DataFrame(np.hstack((np.matmul(valid_X,Z),valid_y.reshape(valid_y.shape[0],1)))).to_csv(save_dir+'/dim_{}/valid.csv'.format(k))
		pd.DataFrame(np.matmul(test_X,Z)).to_csv(save_dir+'/dim_{}/test.csv'.format(k))
		print '[INFO] Data reduced dimesion is {}'.format(k)
	print '[INFO] PCA done!'

train,valid,test,save_dir,red_dim = get_arguments()
train_X,train_y,valid_X,valid_y,test_X = load_data(train,valid,test)
train_X,valid_X,test_X = do_normalizing(train_X,valid_X,test_X)
do_PCA(train_X,train_y,valid_X,valid_y,test_X,save_dir,red_dim)