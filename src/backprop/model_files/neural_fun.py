import copy
import numpy as np
np.random.seed(1234)

# Xavier weight intilization
def weight_init(sizes):
	weights = [np.random.randn(sizes[index],sizes[index+1]) * np.sqrt(1.0/sizes[index]) for index, size in enumerate(sizes) if index != len(sizes)-1]
	biases  = [np.zeros(sizes[index]) for index, size in enumerate(sizes) if index != 0]
	
	return weights,biases

def sigmoid(x):
	 return .5 * (1 + np.tanh(.5 * x))

def sigmoid_diff(x):
	return np.multiply(x,np.subtract(1.0,x))

def Tanh(x):
	return np.tanh(x)

def Tanh_diff(x):
	return np.subtract(1.0,np.square(x))

def relu(x):
	return np.maximum(x, 0)

def relu_diff(x):
	grads  = copy.deepcopy(x)
	grads[grads > 0] = 1
	grads[grads < 0] = 0
	
	return grads

def softMax(x,batch_size):
	x = np.exp(x)
	return np.multiply(x,np.repeat(np.reciprocal(np.sum(x,axis=1)),10).reshape(batch_size,10))

def softMax_diff(x,batch_size,data_y):
	grads  = np.copy(x)
	labels = np.argmax(data_y,axis=1)
	for i in range(batch_size):
		grads[i] = -1*x[i][labels[i]]*grads[i]
		grads[i][labels[i]] += x[i][labels[i]]
	
	return grads

def do_dropout(x,r):
	H = np.copy(x)
	U = (np.random.rand(*H.shape) < r) / float(r)
	
	return U,H*U

def forward_pass(data_X,batch_size,weights,biases,activ_fun,train,data_y=None,dropout=None,dropout_ratio=None):
	layer_inputs  = []
	local_grads   = []
	layer_inputs.append(data_X)
	for index,layer_weights in enumerate(weights):
		output = np.dot(layer_inputs[index],layer_weights) +  np.tile(biases[index],batch_size).reshape(batch_size,biases[index].shape[0])
		if index != len(weights) - 1 : 
			if activ_fun == 'sigmoid':
				output = sigmoid(output)
				if train:
					local_grads.append(sigmoid_diff(output))
					if dropout:
						mask,output = do_dropout(output,dropout_ratio)
						local_grads.append(np.multiply(mask,sigmoid_diff(output)))
			elif activ_fun == 'tanh':
				output = Tanh(output)
				if train:
					local_grads.append(Tanh_diff(output))
					if dropout:
						mask,output = do_dropout(output,dropout_ratio)
						local_grads.append(np.multiply(mask,Tanh_diff(output)))
			else:
				output = relu(output)
				if train:
					local_grads.append(relu_diff(output))
					if dropout:
						mask,output = do_dropout(output,dropout_ratio)
						local_grads.append(np.multiply(mask,relu_diff(output)))					
		else:
			output = softMax(output,batch_size)
			if train:
				local_grads.append(softMax_diff(output,batch_size,data_y))
		layer_inputs.append(output)
	return layer_inputs,local_grads

def get_loss(loss,outputs,data_y,batch_size,weights):
	if loss == 'sq':
		output = np.subtract(outputs,data_y)
		error = np.sum(np.square(output))/float(batch_size)
		output_grad = np.tile(np.sum(output,axis=1),10)
	else:
		outputs_log = np.log2(outputs)  
		error = -1*np.sum(np.multiply(data_y,outputs_log))/float(batch_size)
		labels = np.argmax(data_y,axis=1)
		output_grad = np.reciprocal(-1*outputs)
		for i in range(batch_size):
			output_grad[i] = np.repeat(output_grad[i][labels[i]], 10) 
	
	return error, output_grad

def back_prop(output_grad,local_grads,weights):
	gradients = []
	weights.reverse()
	local_grads.reverse()
	gradients.append(np.multiply(output_grad,local_grads[0]))
	for index,W in enumerate(weights):
		if index != len(weights)-1:
			temp = np.dot(gradients[index],W.T)
			gradients.append(np.multiply(temp,local_grads[index+1]))
	weights.reverse()
	local_grads.reverse()
	gradients.reverse()
	
	return gradients


def weights_update(weights,biases,layer_outputs,gradients,sizes,batch_size,opt,lr,prev_list,anneal_list,momentum,updates,momentum2=0.999,delta=1e-8):
	index = 0
	m_weights,m_biases = [],[]
	v_weights,v_biases = [],[]
	prev_m_weights,prev_m_biases = prev_list[0],prev_list[1]
	prev_v_weights,prev_v_biases = prev_list[2],prev_list[3]
	for grads,W,B,inputs in zip(gradients,weights,biases,layer_outputs):
		inputs_mat = np.vstack([np.tile(inputs[row],sizes[index+1]).reshape(sizes[index+1],sizes[index]).T for row in range(batch_size)])
		grads_mat  = np.vstack([np.tile(grads[row],sizes[index]).reshape(sizes[index],sizes[index+1]) for row in range(batch_size)])
		batch_mat  = np.multiply(inputs_mat,grads_mat)
		delta_W = sum([batch_mat[sizes[index]*j:sizes[index]*(j+1)] for j in range(batch_size)])
		delta_B = np.sum(grads,axis=0)
		if opt == 'gd':
			W_update = -1*lr*delta_W
			B_update = -1*lr*delta_B
		elif opt == 'adam':
			m_weights.append(np.add(momentum*prev_m_weights[index],(1 - momentum)*delta_W))
			m_biases.append(np.add(momentum*prev_m_biases[index],(1 - momentum)*delta_B))
			v_weights.append(np.add(momentum2*prev_v_weights[index],(1 - momentum2)*np.square(delta_W)))
			v_biases.append(np.add(momentum2*prev_v_biases[index],(1 - momentum2)*np.square(delta_B)))
			W_update = -1*lr*(np.sqrt(1 - np.power(momentum2,updates))/(1 - np.power(momentum,updates)))*np.divide(m_weights[index],np.add(np.sqrt(v_weights[index]),delta))
			B_update = -1*lr*(np.sqrt(1 - np.power(momentum2,updates))/(1 - np.power(momentum,updates)))*np.divide(m_biases[index],np.add(np.sqrt(v_biases[index]),delta))
		else:
			W_update = np.subtract(momentum*prev_m_weights[index],lr*delta_W)   
			B_update = np.subtract(momentum*prev_m_biases[index],lr*delta_B)
			m_weights.append(W_update)
			m_biases.append(B_update)		
		W = np.add(W,W_update)
		B = np.add(B,B_update) 
		weights[index] = W
		biases[index] = B
		index = index + 1

	anneal_list = prev_list
	prev_list[0],prev_list[1] = m_weights,m_biases
	prev_list[2],prev_list[3] = v_weights,v_biases
	
	return weights,biases,prev_list,anneal_list