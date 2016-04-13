import os
import ipdb

from fuel import config
from fuel.datasets import H5PYDataset

from fuel.transformers import (Mapping, FilterSources,Flatten,
                            ForceFloatX, ScaleAndShift,Batch)
from fuel.schemes import SequentialScheme
from fuel.schemes import (ShuffledScheme,ConstantScheme,BatchScheme)
from fuel.streams import DataStream

import pickle

class MonkMusic(H5PYDataset):
    def __init__(self, which_sets, filename = 'XqaJ2Ol5cC4.hdf5', **kwargs):
    	self.filename = filename
        super(MonkMusic, self).__init__(self.data_path, which_sets, **kwargs)

    @property
    def data_path(self):
        return os.path.join('/data/lisatmp4/sylvaint/data', 'monk_music', self.filename)

# Getting stats for data
data_dir = os.environ['FUEL_DATA_PATH']
#data_stats_path = os.path.join(data_dir, "monk_music","monk_standardize.npz")
data_stats_path = '/data/lisatmp4/sylvaint/data/monk_music/monk_standardize.npz'
data_stats = pickle.load(open(data_stats_path,"rb"))
print 'data_stats: {}'.format(data_stats)

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _cut_top(data):
    return tuple(array[:, :, :frame_size] for array in data)

# remove swapaxes in this case to conform to keras code
def _get_subsequences(data,batch_size,seq_size,frame_size):
    #return tuple(array.reshape(batch_size,seq_size,frame_size).swapaxes(0,1) for array in data)
    return tuple(array.reshape(batch_size,seq_size,frame_size) for array in data)

def monk_music_stream (which_sets = ('train',),batch_size = 64,
        seq_size=128, frame_size=160, num_examples= None,
        which_sources = ('features',)):

    """
    This function generates the stream for the monk_music dataset.
    It doesn't compute incremental windows and instead simply separates the
    dataset into sequences
    """

    dataset = MonkMusic(which_sets = which_sets, filename = "dataset.hdf5",
        load_in_memory=True)

    large_batch_size = batch_size * frame_size * seq_size
    if not num_examples:
        num_examples = large_batch_size*(dataset.num_examples/large_batch_size)

    # If there are memory problems revert to SequentialScheme
    data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            num_examples,
            large_batch_size))

    data_stream = ScaleAndShift(data_stream,
            scale = 1./data_stats["std"],
            shift = -data_stats["mean"]/data_stats["std"])

    data_stream = Mapping(data_stream,
            lambda data: _get_subsequences(data,batch_size,seq_size,frame_size))

    data_stream = ForceFloatX(data_stream)

    return data_stream
