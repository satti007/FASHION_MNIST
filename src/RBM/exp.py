import os

h_dims  = [400,500,600,700]
samples = [5,10]

k = 1
for n in h_dims:
	os.mkdir('weights_n_{}'.format(n))
	params  = 'python RBM.py --lr 0.01 --h_dim {} --k_step {}'.format(n,k)
	data    = ' --save_dir weights_n_{} --train ../data/train.csv --valid ../data/val.csv --test ../data/test.csv'.format(n)
	cmd = params + data
	print cmd
	os.system(cmd)

print ''

n = 500
for k in samples:
	os.mkdir('weights_k_{}'.format(k))
	params  = 'python RBM.py --lr 0.01 --h_dim {} --k_step {}'.format(n,k)
	data    = ' --save_dir weights_k_{} --train ../data/train.csv --valid ../data/val.csv --test ../data/test.csv'.format(k)
	cmd = params + data
	print cmd
	os.system(cmd)

print ''
