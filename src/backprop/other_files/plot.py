import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LAR = re.findall(r'Getting article_urls from section:(.*?)...',A,re.DOTALL)
Getting article_urls from section:0, gemology...


# getting T_acc,V_acc, T_loss,V_loss for train,valid log_files
def parser(log_train):
    acc, loss = [],[]
    logger = open(log_train,'r')
    data = logger.read()
    LAR = re.findall(r'Step 1000, (.*?), lr:0.0001',data,re.DOTALL)
    for lr in LAR:
        loss.append(float(lr.split(' ')[1].strip(',')))
        acc.append(float(lr.split(' ')[3]))
    
    return acc,loss

# Plot 1: Layers & neurons
layers = [1,2,3,4]
neurons = [50,100,200,300]
log_train_files = [] 
log_val_files = []
for l in layers:
    for h in neurons:
        log_train_files.append('../log_files/log_files_h{}_n{}/log_train.txt'.format(l,h))
        log_val_files.append('../log_files/log_files_h{}_n{}/log_val.txt'.format(l,h))

log_train_files_1 = log_train_files[0:4]
log_train_files_2 = log_train_files[4:8]
log_train_files_3 = log_train_files[8:12]
log_train_files_4 = log_train_files[12:16]

log_val_files_1 = log_val_files[0:4]
log_val_files_2 = log_val_files[4:8]
log_val_files_3 = log_val_files[8:12]
log_val_files_4 = log_val_files[12:16]

train_acc_1,train_loss_1 = parser(log_train_files_1[0])
train_acc_2,train_loss_2 = parser(log_train_files_1[1])
train_acc_3,train_loss_3 = parser(log_train_files_1[2])
train_acc_4,train_loss_4 = parser(log_train_files_1[3])

val_acc_1,val_loss_1 = parser(log_val_files_1[0])
val_acc_2,val_loss_2 = parser(log_val_files_1[1])
val_acc_3,val_loss_3 = parser(log_val_files_1[2])
val_acc_4,val_loss_4 = parser(log_val_files_1[3])

fig = plt.figure()
plt.plot(train_loss_1,label='50')
plt.plot(train_loss_2,label='100')
plt.plot(train_loss_3,label='200')
plt.plot(train_loss_4,label='300')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('train_loss')
fig.savefig('plot_1_train.png')

fig = plt.figure()
plt.plot(val_loss_1,label='50')
plt.plot(val_loss_2,label='100')
plt.plot(val_loss_3,label='200')
plt.plot(val_loss_4,label='300')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('val_loss')
fig.savefig('plot_1_val.png')


# Plot 2: optimizers
opt = ['gd','momentum','nag','adam']
log_train_files = [] 
log_val_files = []
for o in opt:
    log_train_files.append('../log_files/log_files_{}/log_train.txt'.format(o))
    log_val_files.append('../log_files/log_files_{}/log_val.txt'.format(o))

train_acc_1,train_loss_1 = parser(log_train_files[0])
train_acc_2,train_loss_2 = parser(log_train_files[1])
train_acc_3,train_loss_3 = parser(log_train_files[2])
train_acc_4,train_loss_4 = parser(log_train_files[3])

val_acc_1,val_loss_1 = parser(log_val_files[0])
val_acc_2,val_loss_2 = parser(log_val_files[1])
val_acc_3,val_loss_3 = parser(log_val_files[2])
val_acc_4,val_loss_4 = parser(log_val_files[3])

fig = plt.figure()
plt.plot(train_loss_1,label='gd')
plt.plot(train_loss_2,label='momentum')
plt.plot(train_loss_3,label='nag')
plt.plot(train_loss_4,label='adam')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('train_loss')
fig.savefig('plot_opt_train.png')

fig = plt.figure()
plt.plot(val_loss_1,label='gd')
plt.plot(val_loss_2,label='momentum')
plt.plot(val_loss_3,label='nag')
plt.plot(val_loss_4,label='adam')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('val_loss')
fig.savefig('plot_opt_val.png')

# Plot 3: Activation functions
activ_fun = ['sigmoid','tanh']
log_train_files = [] 
log_val_files = []
for f in activ_fun:
    log_train_files.append('../log_files/log_files_{}/log_train.txt'.format(f))
    log_val_files.append('../log_files/log_files_{}/log_val.txt'.format(f))

train_acc_1,train_loss_1 = parser(log_train_files[0])
train_acc_2,train_loss_2 = parser(log_train_files[1])

val_acc_1,val_loss_1 = parser(log_val_files[0])
val_acc_2,val_loss_2 = parser(log_val_files[1])

fig = plt.figure()
plt.plot(train_loss_1,label='sigmoid')
plt.plot(train_loss_2,label='tanh')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('train_loss')
fig.savefig('plot_activ_train.png')

fig = plt.figure()
plt.plot(val_loss_1,label='sigmoid')
plt.plot(val_loss_2,label='tanh')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('val_loss')
fig.savefig('plot_activ_val.png')



# Plot 4: learning rate
lr = [0.01,0.001,0.0001,0.00001]
for l in lr:
    name = 'plot_{}'.format(l)
    parser('../log_files/log_files_{}/log_train.txt'.format(l),'../log_files/log_files_{}/log_val.txt'.format(l),name)

lr_fine = [0.005,0.002,0.001,0.0005,0.00095]
for l in lr_fine:
    name = 'plot_fine_{}'.format(l)
    parser('../log_files/log_files_fine_{}/log_train.txt'.format(l),'../log_files/log_files_fine_{}/log_val.txt'.format(l),name)

train_acc=[]
val_acc=[]
train_loss=[]
val_loss=[]
train = 1
for line in c:
    if '1000' in line:         
        if train:
            train_loss.append(float(line.split(',')[2].split(' ')[2]))
            train_acc.append(float(line.split(',')[3].split(' ')[2]))
            train = 0
        else:
            val_loss.append(float(line.split(',')[2].split(' ')[2]))
            val_acc.append(float(line.split(',')[3].split(' ')[2]))
            train = 1

fig = plt.figure()
plt.plot(train_loss,label='train')
plt.plot(val_loss,label='val')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('loss')
fig.savefig('plot_loss.png')

fig = plt.figure()
plt.plot(train_acc,label='train')
plt.plot(val_acc,label='val')
plt.legend()
plt.title('')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
fig.savefig('plot_acc.png')

# # Plot 3: batch size
# batch_size = [1,20,100,1000]
# log_train_files = [] 
# log_val_files = []
# for b in batch_size:
#     log_train_files.append('../log_files/log_files_{}/log_train.txt'.format(b))
#     log_val_files.append('../log_files/log_files_{}/log_val.txt'.format(b))

# train_acc_1,train_loss_1 = parser(log_train_files[0])
# train_acc_2,train_loss_2 = parser(log_train_files[1])
# train_acc_3,train_loss_3 = parser(log_train_files[2])
# train_acc_4,train_loss_4 = parser(log_train_files[3])

# val_acc_1,val_loss_1 = parser(log_val_files[0])
# val_acc_2,val_loss_2 = parser(log_val_files[1])
# val_acc_3,val_loss_3 = parser(log_val_files[2])
# val_acc_4,val_loss_4 = parser(log_val_files[3])

# fig = plt.figure()
# plt.plot(train_loss_1,label='1')
# plt.plot(train_loss_2,label='20')
# plt.plot(train_loss_3,label='100')
# plt.plot(train_loss_4,label='1000')
# plt.legend()
# plt.title('')
# plt.xlabel('Epochs')
# plt.ylabel('train_loss')
# fig.savefig('plot_batch_train.png')

# fig = plt.figure()
# plt.plot(val_loss_1,label='1')
# plt.plot(val_loss_2,label='20')
# plt.plot(val_loss_3,label='100')
# plt.plot(val_loss_4,label='1000')
# plt.legend()
# plt.title('')
# plt.xlabel('Epochs')
# plt.ylabel('val_loss')
# fig.savefig('plot_batch_val.png')