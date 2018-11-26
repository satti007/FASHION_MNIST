import random
import pickle
import argparse
import numpy as np
import pandas as pd
from neural_fun import *

np.random.seed(1234)

# A function for anneal(T/F) argument
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

# A function for anneal_rate argument to be contained in range (0,1)
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

# Arguments Parser
def get_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('--lr'        ,type=float,default=0.01)
	ap.add_argument('--momentum'  ,type=float,default=0.9)
	ap.add_argument('--max_epochs',type=int,default=20)
	ap.add_argument('--batch_size',type=int,default=1)
	ap.add_argument('--num_hidden',type=int,default=2)
	ap.add_argument('--sizes'     ,type=str,default='100,100')
	ap.add_argument('--zero'   	  ,type=str2bool,default=False)
	ap.add_argument('--norm'      ,type=str2bool,default=False)
	ap.add_argument('--dropout'   ,type=str2bool,default=False)
	ap.add_argument('--anneal'    ,type=str2bool,default=False)
	ap.add_argument('--restore'   ,type=str2bool,default=False)
	ap.add_argument('--earlyStop' ,type=str2bool,default=False)
	ap.add_argument('--anneal_rate'  ,type=restricted_float,default=0.5)
	ap.add_argument('--dropout_ratio',type=restricted_float,default=0.5)
	ap.add_argument('--restore_state',type=int,default=0)
	ap.add_argument('--loss'     ,type=str,default='ce'  ,choices=['sq','ce'])
	ap.add_argument('--activ_fun',type=str,default='relu',choices=['tanh','sigmoid','relu'])
	ap.add_argument('--opt'      ,type=str,default='adam',choices=['gd', 'momentum', 'nag', 'adam'])
	ap.add_argument('--save_dir' ,type=str)
	ap.add_argument('--expt_dir' ,type=str)
	ap.add_argument('--train'    ,type=str)
	ap.add_argument('--valid'    ,type=str)
	ap.add_argument('--test'     ,type=str)
	
	print '[INFO] Parsing the Arguments...'
	args = vars(ap.parse_args())
	
	sizes      = map(int,args['sizes'].split(','))
	num_hidden = args['num_hidden']
	
	if len(sizes) != num_hidden :
		ap.error('len(sizes) is not equal to num of hidden layers')
	if args['batch_size']%5 == 0 or args['batch_size'] == 1 :
		batch_size  = args['batch_size']
	else:
		ap.error('valid values of batch_size are 1 and multiples of 5')
	
	lr         = args['lr']
	opt        = args['opt']
	loss       = args['loss']
	norm       = args['norm']
	zero       = args['zero']
	anneal     = args['anneal']
	restore    = args['restore']
	dropout    = args['dropout']
	momentum   = args['momentum']
	earlyStop  = args['earlyStop']
	max_epochs = args['max_epochs']
	activ_fun  = args['activ_fun']
	restore_state = args['restore_state']
	anneal_rate   = args['anneal_rate']
	dropout_ratio = args['dropout_ratio']
	save_dir  = args['save_dir']
	expt_dir  = args['expt_dir']
	train = args['train']
	valid = args['valid']
	test  = args['test']
	print '[INFO] Arguments Parsing Done!'
	print '[INFO] Arguments details: '
	print 'lr,momentum         : ',lr,momentum
	print 'num_hidden,sizes    : ',num_hidden,sizes
	print 'batch_size,opt,loss : ',batch_size,opt,loss
	print 'activ_fun,max_epochs: ',activ_fun,max_epochs
	print 'restore,restore_state    : ',restore,restore_state
	print 'earlyStop,anneal,dropout : ',earlyStop,anneal,dropout
	print 'anneal_rate,dropout_ratio: ',anneal_rate,dropout_ratio
	print 'train: {}\nvalid: {}\ntest : {}'.format(train,valid,test)
	print 'save_dir: {}\nexpt_dir: {}'.format(save_dir,expt_dir)
	
	return lr,momentum,num_hidden,sizes,batch_size,opt,activ_fun,loss,norm,zero,restore,earlyStop,anneal,dropout,restore_state,anneal_rate,dropout_ratio,max_epochs,save_dir,expt_dir,train,valid,test

# Reading data
def load_data(train,valid,test):
	print '[INFO] Loaidng the data...'
	train_data = pd.read_csv(train).as_matrix()
	valid_data = pd.read_csv(valid).as_matrix()
	test_data  = pd.read_csv(test).as_matrix()
	in_dim     = train_data.shape[1]-2 # assuming there is index at first col & label at the last col
	train_X, train_y = train_data[:,1:in_dim+1], train_data[:,in_dim+1] 
	valid_X, valid_y = valid_data[:,1:in_dim+1], valid_data[:,in_dim+1]                
	test_X  = test_data[:,1:in_dim+1]
	out_dim = np.unique(train_y).shape[0]
	print '[INFO] Train_data details: ',train_X.shape, train_y.shape
	print '[INFO] Valid_data details: ',valid_X.shape, valid_y.shape
	print '[INFO] Test_data details : ',test_X.shape
	print '[INFO] Reading the data Done!'
	
	return train_X,train_y.astype(int),valid_X,valid_y.astype(int),test_X,in_dim,out_dim

# restore the model at the given state(epoch)
def load_weights(save_dir,state):
	with open(save_dir+'/weights_{}.npy'.format(state), 'rb') as f:
		weights = pickle.load(f)
	with open(save_dir+'/biases_{}.npy'.format(state), 'rb') as f:
		biases = pickle.load(f)
	# weights = list(np.load(save_dir+'/weights_{}.npy'.format(state)))
	# biases  = list(np.load(save_dir+'/biases_{}.npy'.format(state)))
	print '[INFO] Model restored at {} Epoch'.format(state)
	
	return weights,biases

# Neural net building and weight intilization 
def build_model(sizes,in_dim,out_dim,restore,restore_state,save_dir):
	print '[INFO] Length of input,output vector: {},{}'.format(in_dim,out_dim)
	print '[INFO] Model and weights intilization...'
	sizes.insert(0,in_dim)
	sizes.append(out_dim)
	if restore:
		weights,biases = load_weights(save_dir,restore_state)
	else:
		weights,biases = weight_init(sizes)
	print '[INFO] Weights,biases details:'
	for i,j in zip(weights,biases):
		print i.shape,j.shape
	print '[INFO] Weights intilization Done!'
	
	return weights,biases

# change range to [0,1] 
def do_normalizing(train_X,valid_X,test_X):
	train_X = train_X/float(np.max(train_X))
	valid_X = valid_X/float(np.max(valid_X))
	test_X  = test_X/float(np.max(test_X))
	
	return train_X,valid_X,test_X

# column-wise mean-centring
def mean_centering(train_X,valid_X,test_X):
	train_X = train_X - np.mean(train_X,axis=0)
	valid_X = valid_X - np.mean(valid_X,axis=0)
	test_X  = test_X  - np.mean(test_X,axis=0)
	
	return train_X,valid_X,test_X

# save weights at the given state(epoch)
def save_weights(save_dir,weights,biases,epochs):
	with open(save_dir+'/biases_{}.npy'.format(epochs),'wb') as fp:
		pickle.dump(biases, fp)
	with open(save_dir+'/weights_{}.npy'.format(epochs),'wb') as fp:
		pickle.dump(weights, fp)
	# np.save(save_dir+'/biases_{}.npy'.format(epochs), biases)
	# np.save(save_dir+'/weights_{}.npy'.format(epochs), weights)
	print '[INFO] Model weights saved at {} epochs'.format(epochs)

# caluclate accuracy
def get_accuracy(outputs,true_labels):
	predictions = np.argmax(outputs,axis=1)
	return np.count_nonzero(predictions == true_labels)/float(predictions.shape[0])

# get batch loss,accuracy
def get_loss_acc(batch_X,batch_y,weights,biases):
	accuracy = 0.0
	loss     = 0.0
	samples  = 10
	rounds   = batch_X.shape[0]/samples
	for i in range (rounds):
		data_X,labels = batch_X[samples*i:samples*(i+1),:],batch_y[samples*i:samples*(i+1)]
		data_y        = np.zeros((samples, 10))
		# print labels[0:samples] 
		data_y[np.arange(samples), labels[0:samples]] = 1
		outputs,temp = forward_pass(data_X,samples,weights,biases,activ_fun,False,data_y)
		error,temp   = get_loss(loss,outputs[-1],data_y,samples,weights)
		accuracy += get_accuracy(outputs[-1],labels)  
		loss     += error
		
	return loss/rounds,accuracy/rounds

# do predictions for test data and save sub file
def get_sub_file(test_X,weights,biases,activ_fun,file_name):
	samples     = 10
	rounds     = test_X.shape[0]/samples
	predictions = np.zeros([test_X.shape[0],1]) 
	for i in range (rounds):
		data_X = test_X[samples*i:samples*(i+1),:]
		outputs,temp = forward_pass(data_X,samples,weights,biases,activ_fun,False)
		predictions[samples*i:samples*(i+1)]  = np.argmax(outputs[-1],axis=1)
	sub_file     = pd.read_csv('../../data/sample_sub.csv')
	sub_file['label'] = predictions
	sub_file.to_csv('../kaggle_subs/{}.csv'.format(sub_file),index = False)

# get the batch data
def get_batch_data(batch_size):
	indicies = random.sample(range(0,train_X.shape[0]), batch_size) 
	batch_X = train_X[indicies]
	batch_y = np.zeros((batch_size, 10))
	batch_y[np.arange(batch_size), train_y[indicies]] = 1
	
	return batch_X,batch_y

# add-on function 
def prev_list_zeros(sizes):
	return 2*[[np.zeros((sizes[index],sizes[index+1])) for index, size in enumerate(sizes) if index != len(sizes)-1],
										 [np.zeros(sizes[index]) for index, size in enumerate(sizes) if index != 0]]

# Training
def do_train(lr,momentum,weights,biases,batch_size,opt,activ_fun,loss,earlyStop,anneal,dropout,anneal_rate,dropout_ratio,max_epochs,train_X,train_y,valid_X,valid_y):
	epochs   = 0
	patience = 0
	anneal_times = 0
	stop_valid_accuracy = 0
	prev_list   = prev_list_zeros(sizes)
	anneal_list = prev_list_zeros(sizes) 
	while(epochs < max_epochs and anneal_times < 5):
		for step in range(0,train_X.shape[0]/batch_size):
			batch_X,batch_y = get_batch_data(batch_size)
			if opt == 'nag':
				weights = [i+momentum*j for i,j in zip(weights,prev_list[0])]
			layer_outputs,local_grads = forward_pass(batch_X,batch_size,weights,biases,activ_fun,True,batch_y,dropout,dropout_ratio)
			error, output_grad        = get_loss(loss,layer_outputs[-1],batch_y,batch_size,weights)
			gradients                 = back_prop(output_grad,local_grads,weights)
			weights,biases,prev_list,anneal_list = weights_update(weights,biases,layer_outputs,gradients,sizes,batch_size,opt,lr,prev_list,anneal_list,momentum,epochs*step+1)
			if step%500 == 0:
				train_loss,train_accuracy = get_loss_acc(train_X,train_y,weights,biases)
				print '[INFO] Epoch {},Step {},T_loss: {},T_acc: {},lr: {}'.format(epochs,step,round(train_loss,5),round(train_accuracy*100,2),lr)			
				train_file.write('Epoch {},Step {},T_loss: {},T_Acc: {},lr: {}\n'.format(epochs,step,round(train_loss,5),round(train_accuracy*100,2),lr))
				valid_loss,valid_accuracy = get_loss_acc(valid_X,valid_y,weights,biases)
				print '[INFO] Epoch {},Step {},V_loss: {},V_acc: {},lr:{}'.format(epochs,step,round(valid_loss,5),round(valid_accuracy*100,2),lr)
				val_file.write('Epoch {},Step {},V_loss: {},V_acc: {},lr:{}\n'.format(epochs,step,round(valid_loss,5),round(valid_accuracy*100,2),lr))
		if earlyStop:
			if valid_accuracy < stop_valid_accuracy:
				patience += 1
				if patience == 5 and not anneal:
					print '[INFO] Early Stopping Model at {}, best_epoch: {}'.format(epochs,epochs-5)
					break
				elif patience == 5 and anneal:
					lr = lr*anneal_rate
					prev_list = anneal_list
					patience = 0
					anneal_times += 1
					weights,biases = load_weights(save_dir,epochs-5)
					print '[INFO] learning_rate annealed by {}, lr: {}'.format(anneal_rate,lr)
					continue
			else:
				patience = 0
				anneal_times = 0
				stop_valid_accuracy = valid_accuracy
		save_weights(save_dir,weights,biases,epochs)
		epochs = epochs + 1

lr,momentum,num_hidden,sizes,batch_size,opt,activ_fun,loss,norm,zero,restore,earlyStop,anneal,dropout,restore_state,anneal_rate,dropout_ratio,max_epochs,save_dir,expt_dir,train,valid,test = get_arguments()
train_X,train_y,valid_X,valid_y,test_X,in_dim,out_dim = load_data(train,valid,test)
if norm:
	train_X,valid_X,test_X = do_normalizing(train_X,valid_X,test_X)
if zero:
	train_X,valid_X,test_X = mean_centering(train_X,valid_X,test_X)
weights,biases = build_model(sizes,in_dim,out_dim,restore,restore_state,save_dir)
print '[INFO] Model training started...'
train_file = open(expt_dir+"/log_train.txt","w",0) 
val_file   = open(expt_dir+"/log_val.txt","w",0) 
do_train(lr,momentum,weights,biases,batch_size,opt,activ_fun,loss,earlyStop,anneal,dropout,anneal_rate,dropout_ratio,max_epochs,train_X,train_y,valid_X,valid_y)
print '[INFO] Model training done!'

