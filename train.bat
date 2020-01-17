py -3.7 main.py --train_ds_file train_rect.pickle --eval_ds_file eval_rect.pickle --noobj_w 0.025 --bb_w 2 --model_type=lenet_5_yolo --lr=1e-4 --model_file=model.pickle --log_file=log.txt --steps=1000 --eval_steps=50 --prev_steps=5000 --learner_file=opt.pickle
copy model.pickle model_checkpoint1.pickle
copy opt.pickle opt_checkpoint1.pickle
py -3.7 main.py --train_ds_file train_rect.pickle --eval_ds_file eval_rect.pickle --noobj_w 0.025 --bb_w 2 --model_type=lenet_5_yolo --lr=1e-4 --model_file=model.pickle --log_file=log.txt --steps=1000 --eval_steps=50 --prev_steps=6000 --learner_file=opt.pickle
copy model.pickle model_checkpoint2.pickle
copy opt.pickle opt_checkpoint2.pickle
py -3.7 main.py --train_ds_file train_rect.pickle --eval_ds_file eval_rect.pickle --noobj_w 0.025 --bb_w 2 --model_type=lenet_5_yolo --lr=1e-4 --model_file=model.pickle --log_file=log.txt --steps=1000 --eval_steps=50 --prev_steps=7000 --learner_file=opt.pickle
copy model.pickle model_checkpoint3.pickle
copy opt.pickle opt_checkpoint3.pickle
py -3.7 main.py --train_ds_file train_rect.pickle --eval_ds_file eval_rect.pickle --noobj_w 0.025 --bb_w 2 --model_type=lenet_5_yolo --lr=1e-4 --model_file=model.pickle --log_file=log.txt --steps=1000 --eval_steps=50 --prev_steps=8000 --learner_file=opt.pickle
copy model.pickle model_checkpoint4.pickle
copy opt.pickle opt_checkpoint4.pickle
py -3.7 main.py --train_ds_file train_rect.pickle --eval_ds_file eval_rect.pickle --noobj_w 0.025 --bb_w 2 --model_type=lenet_5_yolo --lr=1e-4 --model_file=model.pickle --log_file=log.txt --steps=1000 --eval_steps=50 --prev_steps=9000 --learner_file=opt.pickle
