import os

params  = ' --activ_fun relu --loss ce --opt adam --batch_size 50 --max_epochs 20'
flags   = ' --norm    false --zero true --earlyStop false --anneal false'
values  = ' --dropout false --anneal_rate 0.5 --dropout_ratio 0.5'
load    = ' --restore false --restore_state 0'
wts_dir = ' --save_dir ../restore_files/weights'
log_dir = ' --expt_dir ../log_files'
data    = ' --train ../../data/dim_300/train.csv --valid ../../data/dim_300/valid.csv	--test ../../data/dim_300/test.csv'

#PCA experminemt with dimensions
dims = [84,100,200,300,400,500]
for d in dims:
	cmd = 'python -W ignore train.py --lr 0.001 --momentum 0.9 --num_hidden 2 --sizes 300,300'
	cmd = cmd + params + flags + values + load
	os.system('mkdir -p {}/weights_{}'.format(wts_dir.split(' ')[2],d))
	cmd = cmd + wts_dir +'/weights_{}'.format(d)
	os.system('mkdir -p {}/log_files_{}'.format(log_dir.split(' ')[2],d))
	cmd = cmd + log_dir + '/log_files_{}'.format(d)  
	data = ' --train ../../data/dim_{}/train.csv --valid ../../data/dim_{}/valid.csv	--test ../../data/dim_{}/test.csv'.format(d,d,d)
	cmd  = cmd + data
	print cmd + '\n'
	os.system(cmd)

# Experminemt for layers and neurons
layers  = [1,2,3,4]
neurons = [50,100,200,300] 
for h in  layers:
	for n in neurons:
		cmd = 'python -W ignore ../model_files/train.py --lr 0.001 --momentum 0.9 --num_hidden {} --sizes '.format(h)
		for i in range(0,h):
			if i == h-1:
				cmd     = cmd + str(n)
			else:
				cmd     = cmd + str(n) + ',' 
		cmd = cmd + params + flags + values + load
		os.system('mkdir -p {}/weights_{}_{}'.format(wts_dir.split(' ')[2],h,n))
		cmd = cmd + wts_dir +'/weights_{}_{}'.format(h,n)
		os.system('mkdir -p {}/log_files_{}_{}'.format(log_dir.split(' ')[2],h,n))
		cmd = cmd + log_dir + '/log_files_{}_{}'.format(h,n)  
		cmd = cmd + data
		print cmd + '\n'
		os.system(cmd)

# For optimizier
opt = ['gd','momentum','nag','adam']
for o in opt:
	cmd = 'python train.py --lr 0.001 --momentum 0.9 --num_hidden 3 --sizes 300,300,300'
	params  = ' --activ_fun relu --loss ce --opt {} --batch_size 50 --max_epochs 20'.format(o)
	cmd = cmd + params + flags + values + load
	os.system('mkdir -p {}/weights_{}'.format(wts_dir.split(' ')[2],o))
	cmd = cmd + wts_dir +'/weights_{}'.format(o)
	os.system('mkdir -p {}/log_files_{}'.format(log_dir.split(' ')[2],o))
	cmd = cmd + log_dir + '/log_files_{}'.format(o)  
	cmd = cmd + data
	print cmd + '\n'
	os.system(cmd)


# For activation function
activ_fun = ['sigmoid','tanh','relu']
params = ' --loss ce --batch_size 20 --anneal False --opt adam'
for f in activ_fun:
	cmd = 'python train.py --lr 0.001 --momentum 0.9 --num_hidden 3 --sizes 300,300,300'
	params  = ' --activ_fun f --loss ce --opt adam --batch_size 50 --max_epochs 20'.format(f)
	cmd = cmd + params + flags + values + load
	os.system('mkdir -p {}/weights_{}'.format(wts_dir.split(' ')[2],f))
	cmd = cmd + wts_dir +'/weights_{}'.format(f)
	os.system('mkdir -p {}/log_files_{}'.format(log_dir.split(' ')[2],f))
	cmd = cmd + log_dir + '/log_files_{}'.format(f)  
	cmd = cmd + data
	print cmd + '\n'
	os.system(cmd)

# Experminemt for learning rate
lr = [0.01,0.001,0.0001,0.00001]
for l in lr:
	cmd = 'python -W ignore ../model_files/train.py --lr {} --momentum 0.9 --num_hidden 3 --sizes 300,300,300'.format(l)
	cmd = cmd + params + flags + values + load
	os.system('mkdir -p {}/weights_{}'.format(wts_dir.split(' ')[2],l))
	cmd = cmd + wts_dir +'/weights_{}'.format(l)
	os.system('mkdir -p {}/log_files_{}'.format(log_dir.split(' ')[2],l))
	cmd = cmd + log_dir + '/log_files_{}'.format(l)  
	cmd = cmd + data
	print cmd + '\n'
	os.system(cmd)

# Fine-tuning learning rate
lr_fine = [0.005,0.002,0.001,0.0005,0.00095]
for l in lr_fine:
	cmd = 'python train.py --lr {} --momentum 0.9 --num_hidden 3 --sizes 300,300,300'.format(l)
	cmd = cmd + params + flags + values + load
	os.system('mkdir -p {}/weights_{}'.format(wts_dir.split(' ')[2],l))
	cmd = cmd + wts_dir +'/weights_{}'.format(l)
	os.system('mkdir -p {}/log_files_{}'.format(log_dir.split(' ')[2],l))
	cmd = cmd + log_dir + '/log_files_{}'.format(l)  
	cmd = cmd + data
	print cmd + '\n'
	os.system(cmd)