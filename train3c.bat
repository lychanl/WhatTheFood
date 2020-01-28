py -3.7 main.py --train_ds_file train_rect_3c.pickle --eval_ds_file eval_rect_3c.pickle --train_ds_dir data --noobj_w 0.05 --bb_w 2 --model_type=tiny_yolo --lr=1e-4 --steps=20000 --use-tf --eval_steps 100 --learner_file opt3c.pickle --model_file model3c.pickle --log_file log3c.txt --prev_steps 00000 --l2 0.001
copy model.pickle model_checkpoint1.pickle
copy opt.pickle opt_checkpoint1.pickle
py -3.7 main.py --train_ds_file train_rect_3c.pickle --eval_ds_file eval_rect_3c.pickle --train_ds_dir data --noobj_w 0.05 --bb_w 2 --model_type=tiny_yolo --lr=1e-4 --steps=20000 --use-tf --eval_steps 100 --learner_file opt3c.pickle --model_file model3c.pickle --log_file log3c.txt --prev_steps 20000 --l2 0.001
copy model.pickle model_checkpoint2.pickle
copy opt.pickle opt_checkpoint2.pickle
py -3.7 main.py --train_ds_file train_rect_3c.pickle --eval_ds_file eval_rect_3c.pickle --train_ds_dir data --noobj_w 0.05 --bb_w 2 --model_type=tiny_yolo --lr=1e-4 --steps=20000 --use-tf --eval_steps 100 --learner_file opt3c.pickle --model_file model3c.pickle --log_file log3c.txt --prev_steps 40000 --l2 0.001
copy model.pickle model_checkpoint3.pickle
copy opt.pickle opt_checkpoint3.pickle
py -3.7 main.py --train_ds_file train_rect_3c.pickle --eval_ds_file eval_rect_3c.pickle --train_ds_dir data --noobj_w 0.05 --bb_w 2 --model_type=tiny_yolo --lr=1e-4 --steps=20000 --use-tf --eval_steps 100 --learner_file opt3c.pickle --model_file model3c.pickle --log_file log3c.txt --prev_steps 60000 --l2 0.001
copy model.pickle model_checkpoint4.pickle
copy opt.pickle opt_checkpoint4.pickle
py -3.7 main.py --train_ds_file train_rect_3c.pickle --eval_ds_file eval_rect_3c.pickle --train_ds_dir data --noobj_w 0.05 --bb_w 2 --model_type=tiny_yolo --lr=1e-4 --steps=20000 --use-tf --eval_steps 100 --learner_file opt3c.pickle --model_file model3c.pickle --log_file log3c.txt --prev_steps 80000 --l2 0.001
