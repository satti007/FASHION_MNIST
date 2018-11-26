python -W ignore train.py --lr 0.001 --momentum 0.9 --num_hidden 2 --sizes 300,300\
 				--activ_fun relu --loss ce --opt adam --batch_size 50 --max_epochs 20\
				--norm    false --zero true --earlyStop false --anneal false\
				--dropout false --anneal_rate 0.5 --dropout_ratio 0.5\
				--restore false --restore_state 0\
				--save_dir ../restore_files/weights\
				--expt_dir ../log_files\
				--train    ../../data/train.csv\
				--valid    ../../data/valid.csv\
				--test     ../../data/test.csv