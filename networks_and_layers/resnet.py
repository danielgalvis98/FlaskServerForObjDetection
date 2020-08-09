'''
File that contains all the functions that define the components of a Faster-RCNN that has a ResNet50
as base convolutional neural network. Taken and adapted from https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a 
and https://github.com/you359/Keras-FasterRCNN
'''
from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, ZeroPadding2D, Convolution2D, Activation, MaxPooling2D, Add, TimeDistributed, AveragePooling2D, Flatten, Dense
from keras import backend as K
from .FixedBatchNormalization import FixedBatchNormalization
from .RoiPoolingConv import RoiPoolingConv
import os

def get_weight_path():
    '''
    Path to the h5 file with the weights from the official resnet50 weights pre-trained on image-net.
    Downladed from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
    '''
    return os.path.join('weights', 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height) 

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    '''
    A Function that represents an identity block of the ResNet50, as it is defined on the state of art.
    It receives as input the input_tensor, which is the output from the previous layers, the kernel_size of the second convolutional layer
    of the block, the number of filters for the three stages, the stage that this block represents in the full network, which block it is in
    the current stage and if the block will be re-trained or not.

    It outputs a tensor that is the result of applying the input_tensor through the layers of the block.

    This block is supposed to be used on the shared layers of the network.
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1,1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    '''
    Same as the previous block, but Time Distributed. The difference between the normal identity_block is that this one 
    is supposed to be used only on the classification layers, so the input_tensor contains a batch of RoI that are processed independently.

    The output tensor consist of a batch with the result of applying the layers to every RoI on the input_tensor
    '''

    # identity block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2), trainable=True):
    '''
    A Function that represents a convolutional block of the ResNet50, as it is defined on the state of art.
    It receives as input the input_tensor, which is the output from the previous layers, the kernel_size of the second convolutional layer
    of the block, the number of filters for the three stages, the stage that this block represents in the full network, which block it is in
    the current stage, the strides for the first and last convolutional layers and if the block will be re-trained or not.

    It outputs a tensor that is the result of applying the input_tensor through the layers of the block.

    This block is supposed to be used on the shared layers of the network.
    '''

    nb_filter1, nb_filter2, nb_filter3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1,1), strides=strides, name = conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name = conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name = bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1,1), name = conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name = bn_name_base + '2c')(x)
    
    shortcut = Convolution2D(nb_filter3, (1,1), strides=strides, name = conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name = bn_name_base + '1')(shortcut)

    x = Add()([x,shortcut])
    x = Activation('relu')(x)
    return x

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    '''
    Same as the previous block, but Time Distributed. The difference between the normal conv_block is that this one 
    is supposed to be used only on the classification layers, so the input_tensor contains a batch of RoI that are processed independently.

    The output tensor consist of a batch with the result of applying the layers to every RoI on the input_tensor
    '''

    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def nn_base(input_tensor=None, trainable=False):
    '''
    A function that defines the shareable layers of the RPN and the Fast-RCNN. On this case, they are the first layers of the ResNet50.
    As arguments, it can receive an input_tensor that can represent an image, and if this layers will be trained or not. It makes use of
    the identity and convolutional block functions defined previously.

    The output is a tensor that represents the feature map as a result of applying the input_tensor through the network.
    '''

    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    bn_axis = 3

    x = ZeroPadding2D((3,3)) (img_input)

    x = Convolution2D(64, (7,7), strides=(2,2), name='conv1', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1,1), trainable=trainable)
    x = identity_block(x, 3, [64,64,256], stage=2, block='b', trainable = trainable)
    x = identity_block(x, 3, [64,64,256], stage=2, block='c', trainable = trainable)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
    x = identity_block(x, 3, [128,128,512], stage=3, block='b', trainable = trainable)
    x = identity_block(x, 3, [128,128,512], stage=3, block='c', trainable = trainable)
    x = identity_block(x, 3, [128,128,512], stage=3, block='d', trainable = trainable)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable = trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable = trainable)

    return x


def classifier_layers(x, input_shape, trainable=False):
    '''
    Function that defines the Resnet50 layers that are used only by the Fast-RCNN in the classification of the proposed ROI. It is
    composed by the blocks of the last stage of the convolutional network section.

    It receives as arguments the previous layers, which are supposed to outpout a batch of ROI, the input_shape of the RoI and if the layers
    will be re-trained or not.

    This layer makes uses of the Time Distributed blocks and layers, as it is supposed to apply them to every RoI independently.Activation
    
    The output is a tensor that represents a batch of feature maps of every RoI. 
    '''
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2,2), trainable=trainable)

    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7,7)), name='avg_pool')(x)

    return x

def region_proposal_network(base_layers, num_anchors):
    '''
    Function that defines the layers specific to the region proposal networks. It receives as arguments the feature map 
    generated by the base_layers (nn_base) and the number of anchors defined.

    The output consists of an array of 3 tensors: the first one is the classification of every anchor (object or not object), the second one
    the coordinates of the box sorrounding the object of the anchor (if it is an object) after applying a regression and a third one that is
    the same tensor received as input (base_layers).
    '''
    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Convolution2D(num_anchors, (1,1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1,1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    '''
    Function that represents the last stage of the network: the classification of every RoI proposed by the RPN.

    It receives as arguments the tensor of the base_layers(nn_base), the RoIs that will be processed, the number of RoIs, the number of possible
    classes than an object can belong to and if this block will be re-trained or not

    The output consists of an array of 2 tensors: the first one contains, for every RoI, to which class it belongs (what object it is). The second
    one is the result of applying a linear regression to the RoIs, and consists of 4 numbers indicating the coordinates of the box that sorrounds
    the object.
    '''
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)

    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]