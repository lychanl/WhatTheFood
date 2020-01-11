import numpy as np
from datetime import datetime
import functools
import argparse

import whatthefood.data.obj_to_nparray as otn
import whatthefood.data.xml_to_obj as xto
from whatthefood.data.preprocessing import ScalePreprocessor
import whatthefood.nn as nn
from whatthefood.train import SGD, ADAM
import whatthefood.classification.yolo as yolo
from whatthefood.data.preprocessing.mb_samples_preprocessing import AddNoisePreprocessor, FlipPreprocessor
import os
import pickle


def get_data(fname, dname, hcells, wcells, scale):
    if not hcells:
        raise argparse.ArgumentError('--hcells', 'Argument is required if a dataset is created')
    if fname and os.path.isfile(fname):
        print(f'Loading data from file: {fname}')
        with open(fname, 'rb') as f:
            return pickle.load(f)
    elif dname and os.path.isdir(dname):
        print(f'Loading data from dir {dname}')

        anns = xto.parse_dir('data')
        print('Creating dataset')
        ds = otn.get_dataset(anns, hcells, wcells, preprocessing=ScalePreprocessor(scale, np.mean))
        ds.processors = [AddNoisePreprocessor(std=1 / 255, limits=(0, 1)), FlipPreprocessor()]

        if fname:
            print('Saving dataset')
            with open(fname, 'wb') as f:
                pickle.dump(ds, f)

        return ds
    else:
        return None


def get_model(fname, mname, input_shape, output_shape):
    if fname and os.path.isfile(fname):
        print(f'Loading model from file: {fname}')
        with open(fname, 'rb') as f:
            return pickle.load(f)
    elif mname == 'fast_yolo':
        model = yolo.fast_yolo_net(input_shape, output_shape)
        model.initialize_weights(nn.GaussianInitializer())
        return model
    elif mname == 'lenet_5_yolo':
        model = yolo.lenet_5_yolo_net(input_shape, output_shape)
        model.initialize_weights(nn.GaussianInitializer())
        return model
    else:
        raise ValueError('Invalid model specification, failed to create')


def print_eval(ds, name, optimizer, log_file):
    stats = '\t'.join([name] + ['%0.5f' % v for v in optimizer.evaluate(ds)])
    print(stats)
    if log_file:
        log_file.write(stats)
        log_file.write(os.linesep)


def run_optimizer(optimizer, steps, train_ds, eval_ds, log_file, decay, prev_steps, mb_size, ev_steps):
    with np.printoptions(precision=5):
        def log_evals():
            print_eval(train_ds, 'train', optimizer, log_file)
            if eval_ds:
                print_eval(eval_ds, 'eval', optimizer, log_file)

        log_evals()

        for i in range(0, steps, ev_steps):
            for j in range(i, i + ev_steps):
                inp, out = train_ds.get_batch(mb_size)
                dv = 1. / np.sqrt(1. + (prev_steps + j) / 2.) if decay else 1.
                loss = optimizer.run(inp, out, lr_decay=dv)
                step_stats = '\t'.join(map(str, [prev_steps + j + 1, dv * optimizer.lr, loss, datetime.now()]))
                print(step_stats)
                if log_file:
                    log_file.write(step_stats)
                    log_file.write(os.linesep)

            log_evals()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--out_model_file', type=str, default=None)
    parser.add_argument('--model_type', type=str, default=None)

    parser.add_argument('--eval_ds_file', type=str, default=None)
    parser.add_argument('--train_ds_file', type=str, default=None)

    parser.add_argument('--log_file', type=str, default=None)

    parser.add_argument('--eval_ds_dir', type=str, default=None)
    parser.add_argument('--train_ds_dir', type=str, default=None)
    parser.add_argument('--hcells', type=int, required=False, default=12)
    parser.add_argument('--wcells', type=int, required=False, default=None)

    parser.add_argument('--scale', type=int, required=False, default=13)

    parser.add_argument('--learner', type=str, default='ADAM', choices=['ADAM', 'SGD'])
    parser.add_argument('--learner_file', type=str, default=None)

    parser.add_argument('--noobj_w', type=float, default=0.5)
    parser.add_argument('--bb_w', type=float, default=5)

    parser.add_argument('--limit', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=bool, default=False)
    parser.add_argument('--prev_steps', type=int, default=0)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--mb_size', type=int, default=8)
    parser.add_argument('--eval_steps', type=int, default=10)

    args = parser.parse_args()

    train_ds = get_data(args.train_ds_file, args.train_ds_dir, args.hcells, args.wcells, args.scale)
    if not train_ds:
        print('No training dataset given')
        exit(-1)

    eval_ds = get_data(args.eval_ds_file, args.eval_ds_dir, args.hcells, args.wcells, args.scale)

    model = get_model(args.model_file, args.model_type, train_ds.inputs[0].shape, train_ds.outputs[0].shape)

    optimizer = (ADAM if args.learner == 'ADAM' else SGD)(
        model, functools.partial(
            yolo.yolo_loss, noobj_weight=args.noobj_w, bounding_boxes_weight=args.bb_w
        ), limit=args.limit, lr=args.lr
    )

    optimizer.add_metrics(optimizer.loss.inputs[0].inputs)
    optimizer.add_metrics(optimizer.loss.inputs[1])
    optimizer.add_metrics(yolo.yolo_metrics)

    if args.learner_file is not None and os.path.isfile(args.learner_file):
        with open(args.learner_file, 'rb') as lrn_state:
            optimizer.restore(lrn_state)

    log_file = None
    if args.log_file:
        print(f'Logging to {args.log_file}')
        log_file = open(args.log_file, 'a')

    run_optimizer(
        optimizer, args.steps, train_ds, eval_ds,
        log_file, args.decay, args.prev_steps, args.mb_size,
        args.eval_steps
    )

    out_model_file = args.out_model_file if args.out_model_file else args.model_file
    if out_model_file:
        print(f'Saving output to {out_model_file}')
        with open(out_model_file, 'wb') as f:
            pickle.dump(model, f)

    if args.learner_file:
        with open(args.learner_file, 'wb') as lrn_state:
            optimizer.store(lrn_state)

    if log_file:
        log_file.close()
