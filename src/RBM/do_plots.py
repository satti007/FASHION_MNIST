import re
import numpy as np
import matplotlib.pyplot as plt

def parser(log_train):
	logger = open(log_train,'r')
	data   = logger.read()
	train_loss = [float(i) for i in re.findall(r'train_loss: (.*?), valid_loss: ',data,re.DOTALL)]
	val_loss   = [float(i) for i in re.findall(r'valid_loss: (.*?), time:',data,re.DOTALL)]
	
	return train_loss,val_loss

train_loss,val_loss = parser('log_files/params_n.log')
train_loss1,val_loss1 = train_loss[51*0:51*1],val_loss[51*0:51*1]
train_loss2,val_loss2 = train_loss[51*1:51*2],val_loss[51*1:51*2]
train_loss3,val_loss3 = train_loss[51*2:51*3],val_loss[51*2:51*3]
train_loss4,val_loss4 = train_loss[51*3:51*4],val_loss[51*3:51*4]

fig = plt.figure()
plt.plot(train_loss1[1:],label='400')
plt.plot(train_loss2[1:],label='500')
plt.plot(train_loss3[1:],label='600')
plt.plot(train_loss4[1:],label='700')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('train_loss')
fig.savefig('plots/train_n.png')

fig = plt.figure()
plt.plot(val_loss1[1:],label='400')
plt.plot(val_loss2[1:],label='500')
plt.plot(val_loss3[1:],label='600')
plt.plot(val_loss4[1:],label='700')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('valid_loss')
fig.savefig('plots/valid_n.png')

train_loss,val_loss = parser('log_files/params_k.log')
train_loss1,val_loss1 = train_loss[51*0:51*1],val_loss[51*0:51*1]
train_loss3,val_loss3 = train_loss[51*1:51*2],val_loss[51*1:51*2]

fig = plt.figure()
plt.plot(train_loss2[1:],label='1')
plt.plot(train_loss1[1:],label='5')
plt.plot(train_loss3[1:],label='10')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('train_loss')
fig.savefig('plots/train_k.png')

fig = plt.figure()
plt.plot(val_loss2[1:],label='1')
plt.plot(val_loss1[1:],label='5')
plt.plot(val_loss3[1:],label='10')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('valid_loss')
fig.savefig('plots/valid_k.png')