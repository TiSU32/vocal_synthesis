from fuel.datasets.hdf5 import H5PYDataset
import scipy.io.wavfile as wav
from fuel.transformers.sequences import Window
import pickle
import sys
import os
import h5py
import scipy

def wav_to_hdf5(input_filename=None,output_filename=None,stats_filename=None):
    """
    Reads a .wav and writes hdf5. Adapted to the walla dataset.
    """
    if input_filename == None:
        input_filename = "/data/lisatmp4/erraqabi/data/walla/output.wav"

    output_file = h5py.File(output_filename, "w")

    output_dirname = os.path.dirname(output_filename)
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)
    rate,data = scipy.io.wavfile.read(input_filename)
    print "Sample rate for the file: {}".format(rate)
    #output_file = h5py.File(output_filename, "w")

    print data.ndim
    print data.shape
    assert data.ndim == 1

    print "Calculating stats"
    data_stats = {}
    data_stats["mean"] = data.mean()
    data_stats["std"] = data.std()
    pickle.dump(data_stats,open(stats_filename,"wb"))
    print "data_stats: {}".format(data_stats)


    data_length = data.shape[0] #
    #if not data_length % batch_size == 0:

    # split 0.9 0.1 0.1
    #train_valid_length = 160000000
    #index_train = int(0.9 * train_valid_length)
    #index_valid = int(train_valid_length)
    #index_test = int(data_length)

    index_train = int(0.8*data_length)
    index_valid = int(0.9*data_length)
    index_test = data_length

    print "batch indices in order : {}".format((index_train,
                                                index_valid,
                                                index_test))

    data = data.reshape((data_length))

    print "Train example: {}".format(data[index_train-100:index_train])
    print "Valid example: {}".format(data[index_valid-100:index_valid])
    print "Test example: {}".format(data[index_test-100:index_test])


    features = output_file.create_dataset(
        name='features' ,
        shape=data.shape,
        dtype='int16',
        data=data)

    #features.dims[0].label = 'batch'
    #features.dims[0].label = 'time'
    features.dims[0].label = 'feature'

    split_dict = {
        'train': {
            'features' : (0,index_train)},
        'valid': {
            'features' : (index_train + 1,index_valid)},
        'test': {
            'features' : (index_valid + 1,index_test)}
    }

    output_file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    #input_file.close()
    output_file.flush()
    output_file.close()



def build_hdf5_dataset_single_dim(input_filename, output_filename):
    """
    Builds a hdf5 dataset given the input one. The output one will have
    training, valid, and test as sources.
    This function outputs a single dimension for the datasets.
    Adapted to monk_music
    """
    input_file = h5py.File(input_filename, "r")
    output_file = h5py.File(output_filename, "w")

    data = input_file["features"][:]
    data_length = data.shape[1] #
    #if not data_length % batch_size == 0:

    # split 0.9 0.1 0.1
    train_valid_length = 160000000
    index_train = int(0.9 * train_valid_length)
    index_valid = int(train_valid_length)
    index_test = int(data_length)

    print "batch indices in order : {}".format((index_train,
                                                index_valid,
                                                index_test))

    data = data.reshape((data_length))

    print "Train example: {}".format(data[index_train-100:index_train])
    print "Valid example: {}".format(data[index_valid-100:index_valid])
    print "Test example: {}".format(data[index_test-100:index_test])


    features = output_file.create_dataset(
        name='features' ,
        shape=data.shape,
        dtype='int16',
        data=data)

    #features.dims[0].label = 'batch'
    #features.dims[0].label = 'time'
    features.dims[0].label = 'feature'

    split_dict = {
        'train': {
            'features' : (0,index_train)},
        'valid': {
            'features' : (index_train + 1,index_valid)},
        'test': {
            'features' : (index_valid + 1,index_test)}
    }

    output_file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    input_file.close()
    output_file.flush()
    output_file.close()


def build_hdf5_dataset_single_dim(input_filename, output_filename):
    """
    Builds a hdf5 dataset given the input one. The output one will have
    training, valid, and test as sources.
    This function outputs a single dimension for the datasets.
    """
    input_file = h5py.File(input_filename, "r")
    output_file = h5py.File(output_filename, "w")

    data = input_file["features"][:]
    data_length = data.shape[1] #
    #if not data_length % batch_size == 0:

    # split 0.9 0.1 0.1
    train_valid_length = 160000000
    index_train = int(0.9 * train_valid_length)
    index_valid = int(train_valid_length)
    index_test = int(data_length)

    print "batch indices in order : {}".format((index_train,
                                                index_valid,
                                                index_test))

    data = data.reshape((data_length))

    print "Train example: {}".format(data[index_train-100:index_train])
    print "Valid example: {}".format(data[index_valid-100:index_valid])
    print "Test example: {}".format(data[index_test-100:index_test])


    features = output_file.create_dataset(
        name='features' ,
        shape=data.shape,
        dtype='int16',
        data=data)

    #features.dims[0].label = 'batch'
    #features.dims[0].label = 'time'
    features.dims[0].label = 'feature'

    split_dict = {
        'train': {
            'features' : (0,index_train)},
        'valid': {
            'features' : (index_train + 1,index_valid)},
        'test': {
            'features' : (index_valid + 1,index_test)}
    }

    output_file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    input_file.close()
    output_file.flush()
    output_file.close()

def build_hdf5_dataset(input_filename, output_filename,batch_size=64):
    """
    Builds a hdf5 dataset given the input one. The output one will have
    training, valid, and test as sources.
    """
    input_file = h5py.File(input_filename, "r")
    output_file = h5py.File(output_filename, "w")

    data = input_file["features"][:]
    data_length = data.shape[1] #

    #print "Sample from data: {}".format(data[70])
    #if not data_length % batch_size == 0:

    # split 0.9 0.1 0.1
    train_valid_length = 160000000
    batch_index_train = int(0.9 * train_valid_length / float(batch_size))
    batch_index_valid = int(train_valid_length / float(batch_size))
    batch_index_test = int(data_length / float(batch_size))

    print "batch indices in order : {}".format((batch_index_train,
                                                batch_index_valid,
                                                batch_index_test))

    assert(train_valid_length == batch_index_valid * batch_size)

    data = data.reshape(data_length)[:batch_index_test*batch_size]
    data = data.reshape(batch_index_test,batch_size,1)
    print data.shape

    print ("values lost: {}").format(data_length - data.size)
    test_length = data_length - train_valid_length

    features = output_file.create_dataset(
        name='features' ,
        shape=data.shape,
        dtype='int16',
        data=data)

    features.dims[0].label = 'batch'
    features.dims[1].label = 'time'
    features.dims[2].label = 'feature'

    split_dict = {
        'train': {
            'features' : (0, batch_index_train)},
        'valid': {
            'features' : (batch_index_train + 1, batch_index_valid)},
        'test': {
            'features' : (batch_index_valid + 1,batch_index_test)}
    }

    output_file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    input_file.close()
    output_file.flush()
    output_file.close()

def test_hdf5_dataset(file_name):
    input_file = h5py.File(file_name, "r")
    print H5PYDataset(which_sets = ('train',), file_or_path = file_name,
        load_in_memory=True).num_examples

    print H5PYDataset(which_sets = ('test',), file_or_path = file_name,
        load_in_memory=True).num_examples

    print H5PYDataset(which_sets = ('valid',), file_or_path = file_name,
        load_in_memory=True).num_examples
    input_file.close()

if __name__=="__main__":
    data_dir = "/data/lisatmp4/sylvaint/data/"
    #input_filename = os.path.join(data_dir,"monk_music","XqaJ2Ol5cC4.hdf5")
    input_filename = "/data/lisatmp4/erraqabi/data/walla/output.wav"
    output_filename = os.path.join(data_dir,"walla","walla_dataset.hdf5")
    stats_filename = os.path.join(data_dir, "walla","walla_standardize.npz")
    #batch_size = 64

    print "Building dataset"
    #build_hdf5_dataset_single_dim(input_filename,output_filename)
    wav_to_hdf5(input_filename,output_filename,stats_filename)
    print "Done"

    print "Testing"
    test_hdf5_dataset(output_filename)
