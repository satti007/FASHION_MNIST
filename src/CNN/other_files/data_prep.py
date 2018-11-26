import random
import argparse
import numpy as np
import pandas as pd 

def get_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('--lr',         type=float, default=0.001)
	ap.add_argument('--batch_size', type=int, default=1)
	ap.add_argument('--init',       type=int, default=1)
	ap.add_argument('--save_dir', type=str)

	print '[INFO] Parsing the Arguments...'
	args = vars(ap.parse_args())
	
	if args['batch_size']%5 == 0 or args['batch_size'] == 1 :
		batch_size  = args['batch_size']
	else:
		ap.error('valid values are 1 and multiples of 5')
	
	lr   = args['lr']
	init = args['init']
	save_dir  = args['save_dir']
	print '[INFO] Arguments Parsing Done!'
	print '[INFO] Arguments details: '
	print 'lr,batch_size,init: ',lr,batch_size,init
	print 'save_dir: ',save_dir
	
	return lr,batch_size,init,save_dir

# Reading data
def load_data(train,valid,test):
	print '[INFO] Loading the data...'
	train_data = pd.read_csv(train).as_matrix()
	valid_data = pd.read_csv(valid).as_matrix()
	test_data = pd.read_csv(test).as_matrix()
	train_X, train_y = train_data[:,1:785], train_data[:,785]
	valid_X, valid_y = valid_data[:,1:785], valid_data[:,785]
	test_X = test_data[:,1:785]
	
	train_X = train_X/255.0
	valid_X = valid_X/255.0
	test_X  = test_X/255.0
	
	train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
	valid_X = valid_X.reshape(valid_X.shape[0], 28, 28, 1)
	test_X  = test_X.reshape(test_X.shape[0], 28, 28, 1)
	
	print '[INFO] Training_data details: ',train_X.shape, train_y.shape
	print '[INFO] Validation_data details: ',valid_X.shape, valid_y.shape
	print '[INFO] Testing_data details: ',test_X.shape
	print '[INFO] Reading the data Done!'
	
	return train_X, train_y,valid_X, valid_y,test_X

# get the batch data
def get_batch_data(batch_size,train_X,train_y,valid_X,valid_y,step,isTrain):
	if isTrain:
		data_X = train_X
		data_y = train_y
		indicies = list(np.arange(batch_size*(step-1),batch_size*step)) 
	else:
		data_X = valid_X
		data_y = valid_y
		indicies = list(np.arange(batch_size*(step-1),batch_size*step)) 
	
	batch_X = data_X[indicies]
	batch_y = np.zeros((batch_size, 10))
	batch_y[np.arange(batch_size), data_y[indicies]] = 1
	
	return batch_X,batch_y

# Data Augmentation -- flip_lr,flip_ud,transpose 
def get_aug_data(X_train,y_train):
	X_train_flip_lr = np.fliplr(X_train)
	X_train_flip_ud = np.flipud(X_train)
	X_train_trans   = np.transpose(X_train,axes=(0,2,1,3))
	y_train_flip_lr = np.copy(y_train)
	y_train_flip_ud = np.copy(y_train)
	y_train_trans   = np.copy(y_train)
	
	X_train = np.vstack([X_train,X_train_flip_lr,X_train_flip_ud,X_train_trans])
	y_train = np.vstack([y_train,y_train_flip_lr,y_train_flip_ud,y_train_trans])
	# print('New data shapes after augmentation:')
	# print('X_train shape: ', X_train.shape)
	# print('y_train shape: ', y_train.shape)
	
	indicies = random.sample(range(0,X_train.shape[0]),X_train.shape[0]) # shuffle the train data
	return X_train[indicies], y_train[indicies]

# train_X = np.zeros([1,28,28,1])
# train_y = np.array([0,1,0,0,0,0,0,0,0,0])
# X_train, y_train = get_aug_data(train_X,train_y)