# encoding:utf-8
import tensorflow as tf
import numpy as np
import os, argparse, time, random
from code.multi_dnn.model_dnn import mul_dnn
from code.util.base_util import get_logger
from code.multi_dnn.nn_prepare_data import read_corpus
from code.base_data_process import load_label2index

# session configuration
config = tf.ConfigProto()
# hyperparameters
parser = argparse.ArgumentParser(description='mul-dnn for recommendation')
parser.add_argument('--train_data', type=str, default='../../origin_data', help='train data source')
parser.add_argument('--test_data', type=str, default='../../origin_data', help='test data source')
parser.add_argument('--result', type=str, default='../../result', help='test data source')
parser.add_argument('--batch_size', type=int, default=128, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=25, help='#epoch of training')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--beta', type=float, default=0.001, help='l2 beta')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--hidden_dim1', type=int, default=60, help='#dim of hidden state')
parser.add_argument('--hidden_dim2', type=int, default=30, help='#dim of hidden state')
parser.add_argument('--hidden_dim3', type=int, default=10, help='#dim of hidden state')
parser.add_argument('--tag', type=str, default='dnn', help='#the tag of the model')

# change mode when test 1
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='dnn_1538797636', help='model for test and demo')
args = parser.parse_args()
## paths setting
paths = {}
tag_timestamp = '{}_{}'.format(args.tag, int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join(args.result, "dnn_save", tag_timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

_, label_list = load_label2index()
print(label_list)
# training model
train_path = os.path.join(args.train_data, 'train_modified.csv')
test_path = os.path.join(args.test_data, 'test_modified.csv')
if args.mode == 'train':
    ids, train_data = read_corpus(train_path)
    print("train data: {}".format(len(train_data)))
    train = train_data[:1000]
    val = train_data[1000:1050]
    input_size = len(train.columns) - 1
    print('input_size', input_size)
    model = mul_dnn(args, len(label_list), input_size, paths, config=config)
    model.build_graph()
    model.train(train=train, dev=val)
elif args.mode == 'test':
    ids, feats_test = read_corpus(test_path)
    print("test data: {}".format(len(feats_test)))
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    input_size = len(feats_test.columns)
    print('input_size', input_size)
    model = mul_dnn(args, len(label_list), input_size, paths, config=config)
    model.build_graph()
    model.test(ids, feats_test)
else:
    print('invalid mode parameter')
