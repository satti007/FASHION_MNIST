import re
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

mean_BN = np.load('models/model_BN/logits_mean.npy')
mean_no_BN = np.load('models/model_no_BN/logits_mean.npy')

fig = plt.figure()
plt.plot(mean_BN)
plt.title('With BN')
plt.xlabel('steps')
plt.ylabel('mean')
plt.savefig('plots/BN.png' )
plt.close()

fig = plt.figure()
plt.plot(mean_no_BN)
plt.title('Without BN')
plt.xlabel('steps')
plt.ylabel('mean')
plt.savefig('plots/no_BN.png' )
plt.close()

def plot_model_history(train_loss,train_acc,val_loss,val_acc,file_name):
	fig, axs = plt.subplots(1,2,figsize=(15,5))
	# summarize history for accuracy
	axs[0].plot(range(1,len(train_acc)+1),train_acc)
	axs[0].plot(range(1,len(val_acc)+1),val_acc)
	axs[0].set_title('Model Accuracy')
	axs[0].set_ylabel('Accuracy')
	axs[0].set_xlabel('Epoch')
	axs[0].set_xticks(np.arange(1,len(train_acc)+1),len(train_acc)/10)
	axs[0].legend(['train', 'val'], loc='best')
	# summarize history for loss
	axs[1].plot(range(1,len(train_loss)+1),train_loss)
	axs[1].plot(range(1,len(val_loss)+1),val_loss)
	axs[1].set_title('Model Loss')
	axs[1].set_ylabel('Loss')
	axs[1].set_xlabel('Epoch')
	axs[1].set_xticks(np.arange(1,len(train_loss)+1),len(train_loss)/10)
	axs[1].legend(['train', 'val'], loc='best')
	fig.savefig(file_name)

def parser(log_train):
	logger = open(log_train,'r')
	data = logger.read()
	train_loss = [float(i) for i in re.findall(r'Total_train_epoch_loss: (.*?), Total_train_epoch_acc:',data,re.DOTALL)]
	train_acc  = [float(i) for i in re.findall(r'Total_train_epoch_acc:(.*?)\n',data,re.DOTALL)]
	val_loss   = [float(i) for i in re.findall(r'Total_valid_epoch_loss: (.*?), Total_valid_epoch_acc:',data,re.DOTALL)]
	val_acc    = [float(i) for i in re.findall(r'Total_valid_epoch_acc:(.*?)\n',data,re.DOTALL)]
	
	return train_loss,train_acc,val_loss,val_acc

train_loss,train_acc,val_loss,val_acc = parser('best_model/train.log')
plot_model_history(train_loss,train_acc,val_loss,val_acc,'train_val_loss_acc.png')

train_loss,train_acc,val_loss,val_acc = parser('models/model_BN/train.log')
train_loss1,train_acc1,val_loss1,val_acc1 = parser('models/model_no_BN/train.log')

fig = plt.figure()
plt.plot(train_loss,label='With BN')
plt.plot(train_loss1,label='Without BN')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('train_loss')
fig.savefig('plots/plot_BN_loss.png')

fig = plt.figure()
plt.plot(train_acc,label='With BN')
plt.plot(train_acc1,label='Without BN')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('train_acc')
fig.savefig('plots/plot_BN_acc.png')

train_loss,train_acc,val_loss,val_acc = parser('models/model_BN/train.log')
train_loss1,train_acc1,val_loss1,val_acc1 = parser('models/model_BN_20/train_20.log')
train_loss4,train_acc4,val_loss4,val_acc4 = parser('models/model_BN_100/train.log')
train_loss2,train_acc2,val_loss2,val_acc2 = train_loss4[:21],train_acc4[:21],val_loss4[:21],val_acc4[:21]
train_loss3,train_acc3,val_loss3,val_acc3 = train_loss4[21:],train_acc4[21:],val_loss4[21:],val_acc4[21:]

fig = plt.figure()
plt.plot(train_loss1,label='20')
plt.plot(train_loss,label='50')
plt.plot(train_loss2,label='100')
plt.plot(train_loss3,label='1000')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('train_loss')
fig.savefig('plots/plot_batch_train.png')

fig = plt.figure()
plt.plot(val_acc1,label='20')
plt.plot(val_acc,label='50')
plt.plot(val_acc2,label='100')
plt.plot(val_acc3,label='1000')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('val_acc')
fig.savefig('plots/plot_batch_val.png')


train_loss,train_acc,val_loss,val_acc = parser('models/model_BN/train.log')
train_loss1,train_acc1,val_loss1,val_acc1 = parser('models/model_BN_0.001/train.log')
train_loss2,train_acc2,val_loss2,val_acc2 = parser('models/model_BN_0.0001/train.log')
train_loss3,train_acc3,val_loss3,val_acc3 = parser('models/model_BN_0.00001/train.log')
train_loss4,train_acc4,val_loss4,val_acc4 = parser('models/model_BN_0.0003/train.log')

fig = plt.figure()
plt.plot(train_loss1,label='0.001')
plt.plot(train_loss,label='0.0005')
plt.plot(train_loss4,label='0.0003')
plt.plot(train_loss2,label='0.0001')
plt.plot(train_loss3,label='0.00001')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('train_loss')
fig.savefig('plots/plot_lr_train.png')

fig = plt.figure()
plt.plot(val_acc1,label='0.001')
plt.plot(val_acc, label='0.0005')
plt.plot(val_acc4,label='0.0003')
plt.plot(val_acc2,label='0.0001')
plt.plot(val_acc3,label='0.00001')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('val_acc')
fig.savefig('plots/plot_lr_val.png')