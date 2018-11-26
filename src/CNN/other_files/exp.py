import os

weights_dir = ' --save_dir /home/ubuntu/PA3/src/'
data = ' --train /home/ubuntu/PA3/data/train.csv --val /home/ubuntu/PA3/data/val.csv'\
	   ' --test /home/ubuntu/PA3/data/test.csv'

# batch_size = [20,50,100,1000]
# for b in batch_size:
# 	cmd = 'python train_aws.py --lr 0.0005 --batch_size {} --init 1'.format(b)
# 	os.mkdir(weights_dir.split(' ')[2] + 'weights_{}'.format(b))
# 	cmd = cmd + weights_dir
# 	cmd = cmd + 'weights_{}'.format(b) + data
# 	print cmd
# 	os.system(cmd)

# lr = [0.001,0.0001,0.00001,0.0003]
lr = [0.0001]
for l in lr:
	cmd = 'python train_aws.py --lr {} --batch_size 100 --init 1'.format(l)
	os.mkdir(weights_dir.split(' ')[2] + 'weights_{}'.format(l))
	cmd = cmd + weights_dir
	cmd = cmd + 'weights_{}'.format(l) + data
	print cmd
	# os.system(cmd)
