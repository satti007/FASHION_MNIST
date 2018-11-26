import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

RS = 1234

def do_tSNE(data_X,k_step,h_dim):
	print '[INFO] Initial-Shape of data : ',data_X.shape
	start_time = time.time()
	proj_2d = TSNE(random_state=RS).fit_transform(data_X)
	print '[INFO]     tSNE-Shape of data : ',proj_2d.shape
	np.save('proj_2d/val_{}_{}.npy'.format(h_dim,k_step), proj_2d)
	print '[INFO] tSNE reps of valid_data are saved in: proj.2d/val_{}_{}.npy'.format(h_dim,k_step)
	print 'Time taken:{}'.format(round((time.time() - start_time),2))