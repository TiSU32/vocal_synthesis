import ipdb
import numpy
import theano
import matplotlib
import os
import sys
matplotlib.use('Agg')

from theano import tensor, config, function

import pickle
from config import Config

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

from vocal_synthesis.datasets.monk_music import monk_music_stream


# Reading configuration for the model
with file('/data/lisatmp4/sylvaint/vocal_synthesis/experiments/mm_small.cfg') as f:
    cfg = Config(f)

print cfg
assert(cfg.frame_size % cfg.sub_frame_size == 0)
floatX = theano.config.floatX

# Getting name, creating output directory
print 'save_dir: {}'.format(cfg.save_dir)
directories = [os.path.join(cfg.save_dir,'progress/'),\
    os.path.join(cfg.save_dir,'pkl/')]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
print 'directories {}'.format(directories)

# Creating training and validation streams
train_stream = monk_music_stream(
        which_sets = ('train',),
        batch_size=cfg.batch_size,
        seq_size=cfg.seq_size,
        frame_size=cfg.frame_size,
        num_examples= None,
        which_sources = ('features',))
print "Created train set"

valid_stream = monk_music_stream(
        which_sets = ('valid',),
        batch_size=cfg.batch_size,
        seq_size=cfg.seq_size,
        frame_size=cfg.frame_size,
        num_examples= None,
        which_sources = ('features',))
print "Created validation stream"

# Defining model
model = Sequential()
input_shape = (cfg.batch_size,cfg.seq_size,cfg.frame_size)
# adding input mlp to the model
for i in range(1):
    model.add(Dense(
        output_dim = cfg.dim_mlp,
        batch_input_shape = input_shape,
        init='glorot_uniform',
        activation='relu',
        weights=None,
        W_regularizer=None,
        b_regularizer=None,
        activity_regularizer=None,
        W_constraint=None,
        b_constraint=None))
    model.add(Dropout(0.2))

# adding LSTM layers to the model
#lstm_input_dim = mlp_input_dim
for i in range(1):
    model.add(LSTM(
        output_dim = cfg.dim_lstm,
        init='glorot_uniform',
        inner_init='orthogonal',
        forget_bias_init='one',
        activation='tanh',
        inner_activation='hard_sigmoid',
        W_regularizer=None,
        U_regularizer=None,
        b_regularizer=None,
        dropout_W=0.0,
        dropout_U=0.0))

model.compile(
    optimizer='rmsprop',
    loss="mean_squared_error",
    metrics=['accuracy'])

X = np.zeros((3,cfg.batch_size,cfg.seq_size,cfg.frame_size))
y = np.zeros((3,cfg.batch_size,1))

model.fit(X,y,num_epochs=3)
print model.evaluate(X,y)
