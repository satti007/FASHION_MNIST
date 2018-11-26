import time
import random
import argparse
import numpy as np
import pandas as pd
from tSNE import *
from GSampling import *
np.random.seed(1234)

log = open('params.log','a',0)

# Arguments Parser
def get_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('--lr',    type=float, default=0.01)
	ap.add_argument('--h_dim', type=int,default=100)
	ap.add_argument('--k_step',type=int,default=1)
	ap.add_argument('--train', type=str)
	ap.add_argument('--valid', type=str)
	ap.add_argument('--test' , type=str)
	ap.add_argument('--save_dir', type=str)
	
	print '[INFO] Parsing the Arguments...'
	args = vars(ap.parse_args())
	
	lr     = args['lr']
	h_dim  = args['h_dim']
	k_step = args['k_step'] 
	train  = args['train']
	valid  = args['valid']
	test   = args['test']
	save_dir  = args['save_dir']
	print '[INFO] Arguments Parsing Done!'
	print '[INFO] Arguments details: '
	print 'lr,h_dim,k_step: ',lr,h_dim,k_step
	log.write('lr,h_dim,k_step:{},{},{}\n'.format(lr,h_dim,k_step))
	print 'train: {}\ntest: {}\nvalid: {}\nsave_dir: {}'.format(train,valid,test,save_dir)
	log.write('train: {},test: {}\nvalid: {}\nsave_dir: {}\n'.format(train,valid,test,save_dir))
	
	return lr,h_dim,k_step,train,valid,test,save_dir

def load_data(train,valid,test):
	print '[INFO] Loaidng the data...'
	train_data = pd.read_csv(train).as_matrix()
	valid_data = pd.read_csv(valid).as_matrix()
	test_data  = pd.read_csv(test).as_matrix()
	train_X, train_y = train_data[:,1:785], train_data[:,785]
	valid_X, valid_y = valid_data[:,1:785], valid_data[:,785]
	test_X = test_data[:,1:785]
	
	train_X = train_X/127
	valid_X = valid_X/127
	test_X  = test_X/127
	
	train_X[train_X==2] = 1
	valid_X[valid_X==2] = 1
	test_X[test_X==2]   = 1
	
	print '[INFO] Training_data   details: ',train_X.shape, train_y.shape
	print '[INFO] Validation_data details: ',valid_X.shape, valid_y.shape
	print '[INFO] Testing_data    details: ',test_X.shape
	print '[INFO] Reading the data Done!'
	
	return train_X,train_y, valid_X,valid_y, test_X

# weights, biases intilization
def build_model(v_dim,h_dim,p_train):
	weights   = np.random.normal(scale=0.01, size=(h_dim,v_dim))
	h_biases  = np.zeros([h_dim,1])
	v_biases  = np.zeros([v_dim,1])
	# v_biases  = np.divide(p_train,1 - p_train)
	# v_biases[v_biases == 0] = 1
	# v_biases  = np.log2(v_biases)
	
	return weights,v_biases,h_biases

# Neural net building and weight intilization 
def get_model(train_X,h_dim):
	v_dim = train_X.shape[1]
	print '[INFO] Length of visible vector: ', v_dim
	print '[INFO] Length of hidden vector: ', h_dim
	print '[INFO] Model and weights intilization...'
	p_train = (np.count_nonzero(train_X,axis=0).reshape(v_dim,1))/float(train_X.shape[0])
	weights,v_biases,h_biases = build_model(v_dim,h_dim,p_train)
	print '[INFO] Weights details: {}'.format(weights.shape)
	print '[INFO] v_biases: {},h_biases:{}'.format(v_biases.shape,h_biases.shape)
	print '[INFO] Weights intilization Done!'
	
	return weights,v_biases,h_biases

# load the weights at a given state
def load_weights(save_dir,state):
	weights = np.load(save_dir+'/weights_{}.npy'.format(state))
	biases  = list(np.load(save_dir+'/biases_{}.npy'.format(state)))
	v_biases,h_biases = biases[0],biases[1]
	print '[INFO] Model restored at {} epoch'.format(state)
	
	return weights,v_biases,h_biases

# save the weights in .npy format
def save_weights(save_dir,weights,v_biases,h_biases,epochs):
	print ' | weights saved at {} epoch'.format(epochs)
	np.save(save_dir+'/weights_{}.npy'.format(epochs), weights)
	np.save(save_dir+'/biases_{}.npy'.format(epochs),  [v_biases,h_biases])

# sampling
def do_sampling(save_dir,state,k_step,v_dim,h_dim,data_X):
	hidden_reps = np.zeros([data_X.shape[0],h_dim])
	weights,v_biases,h_biases = load_weights(save_dir,state)
	for i in range(0,data_X.shape[0]):
		V_d = data_X[i].reshape(v_dim,1)
		for j in range(0,k_step):
			H_t = gibbs_sample_RBM(weights,h_biases,v_biases,False,True,V_d)
			V_T = gibbs_sample_RBM(weights,h_biases,v_biases,True,False,None,H_t)
			V_d = np.copy(V_T)
		hidden_reps[i] = np.squeeze(np.asarray(H_t))
	print '[INFO] Hidden_reps for valid_data are of shape:{}'.format(hidden_reps.shape)
	do_tSNE(hidden_reps,k_step,h_dim)

# sigmoid function
def sigmoid(x):
	 return .5 * (1 + np.tanh(.5 * x))

# parameters updates 
def weights_update(weights,h_biases,v_biases,V_d,V_T,H_t,lr):
	weights  += lr*(np.dot(sigmoid(np.dot(weights,V_d)+h_biases),V_d.T)-np.dot(sigmoid(np.dot(weights,V_T)+h_biases),V_T.T))
	v_biases += lr*(V_d - V_T)
	h_biases += lr*(sigmoid(np.dot(weights,V_d)+h_biases) - sigmoid(np.dot(weights,V_T)+h_biases))
	
	return weights,v_biases,h_biases

def get_SQE(V_d,V_T):
	return np.mean(np.square(V_d - V_T))

# reconstruction error -- MSE
def get_loss(weights,v_biases,h_biases,k_step,v_dim,train_X,valid_X):
	T_loss,V_loss = 0,0
	
	for i in range(0,train_X.shape[0]):
		V_d    = train_X[i].reshape(v_dim,1)
		V_init = V_d
		for j in range(0,k_step):
			H_t = gibbs_sample_RBM(weights,h_biases,v_biases,False,True,V_init)
			V_T = gibbs_sample_RBM(weights,h_biases,v_biases,True,False,None,H_t)
			V_init = np.copy(V_T)
		
		T_loss = T_loss + get_SQE(V_d,V_T)
	
	for i in range(0,valid_X.shape[0]):
		V_d    = valid_X[i].reshape(v_dim,1)
		V_init = V_d
		for j in range(0,k_step):
			H_t = gibbs_sample_RBM(weights,h_biases,v_biases,False,True,V_init)
			V_T = gibbs_sample_RBM(weights,h_biases,v_biases,True,False,None,H_t)
			V_init = np.copy(V_T)
		
		V_loss = V_loss + get_SQE(V_d,V_T)
	
	return float(T_loss)/(train_X.shape[0]),float(V_loss)/(valid_X.shape[0])

# Training 
def do_train(max_epochs,save_dir,weights,v_biases,h_biases,k_step,lr,v_dim,h_dim,train_X,valid_X):
	epochs,tol    = 1,100
	best_epoch,best_val_loss = 1,100
	T_loss,V_loss = get_loss(weights,v_biases,h_biases,k_step,v_dim,train_X,valid_X)
	start_time = time.time()
	print '[INFO] Epoch 0, train_loss: {}, valid_loss: {}, time:{}'.format(round(T_loss,5),round(V_loss,5),round((time.time() - start_time),2))
	log.write('[INFO] Epoch 0, train_loss: {}, valid_loss: {}, time:{}\n'.format(round(T_loss,5),round(V_loss,5),round((time.time() - start_time),2)))
	while(epochs < max_epochs+1 and tol > 1e-3):
		start_time = time.time()
		for i in range(0,train_X.shape[0]):
			V_d    = train_X[i].reshape(v_dim,1)
			V_init = V_d
			for j in range(0,k_step):
				H_t = gibbs_sample_RBM(weights,h_biases,v_biases,False,True,V_init)
				V_T = gibbs_sample_RBM(weights,h_biases,v_biases,True,False,None,H_t)
				V_init = np.copy(V_T)
			weights,v_biases,h_biases = weights_update(weights,h_biases,v_biases,V_d,V_T,H_t,lr)
		T_loss,V_loss = get_loss(weights,v_biases,h_biases,k_step,v_dim,train_X,valid_X)
		print '[INFO] Epoch {} train_loss: {}, valid_loss: {}, time:{}'.format(epochs,round(T_loss,5),round(V_loss,5),round((time.time() - start_time),2)),
		log.write('[INFO] Epoch {} train_loss: {}, valid_loss: {}, time:{}\n'.format(epochs,round(T_loss,5),round(V_loss,5),round((time.time() - start_time),2)))
		save_weights(save_dir,weights,v_biases,h_biases,epochs)
		if V_loss <= best_val_loss:
			best_epoch,best_val_loss = epochs,V_loss
		tol    = T_loss
		epochs = epochs + 1
	
	return best_epoch,best_val_loss

lr,h_dim,k_step,train,valid,test,save_dir = get_arguments()
train_X,train_y,valid_X,valid_y,test_X    = load_data(train,valid,test)
weights,v_biases,h_biases = get_model(train_X,h_dim)
max_epochs = 50
print '[INFO] Model training started...'
best_epoch,best_val_loss = do_train(max_epochs,save_dir,weights,v_biases,h_biases,k_step,lr,train_X.shape[1],h_dim,train_X,valid_X)
print '[INFO] Model training done!'
print '[INFO] valid_loss is min at:{} epoch of value:{}'.format(best_epoch,round(best_val_loss,5))
print '[INFO] Getting hidden_reps for valid_data..'
do_sampling(save_dir,best_epoch,k_step,train_X.shape[1],h_dim,valid_X)
print '[INFO] Done!'