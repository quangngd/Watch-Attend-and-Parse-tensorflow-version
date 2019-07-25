import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
import numpy as np
from data import dataIterator, load_dict, prepare_data
from compute_wer import process as wer_process
import random
import sys
import copy
import copy
import re
import os
import time
import math
import argparse

rng = np.random.RandomState(int(time.time()))

# for that one assert line out of nowhere?
dictLen = None

"""
following three functions:
norm_weight(),
conv_norm_weight(),
ortho_weight()
are initialization methods for weights
"""


def norm_weight(fan_in, fan_out):
    W_bound = np.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(
        rng.uniform(low=-W_bound, high=W_bound, size=(fan_in, fan_out)),
        dtype=np.float32,
    )


def conv_norm_weight(nin, nout, kernel_size):
    filter_shape = (kernel_size[0], kernel_size[1], nin, nout)
    fan_in = kernel_size[0] * kernel_size[1] * nin
    fan_out = kernel_size[0] * kernel_size[1] * nout
    W_bound = np.sqrt(6.0 / (fan_in + fan_out))
    W = np.asarray(
        rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=np.float32
    )
    return W.astype("float32")


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype("float32")


class Watcher_train:
    def __init__(
        self,
        blocks,  # number of dense blocks
        level,  # number of levels in each blocks
        growth_rate,  # growth rate in DenseNet paper: k
        training,
        B_option,
        dropout_rate=0.2,  # keep-rate of dropout layer
        dense_channels=0,  # filter numbers of transition layer's input
        transition=0.5,  # rate of comprssion
        input_conv_filters=48,  # filter numbers of conv2d before dense blocks
        input_conv_stride=2,  # stride of conv2d before dense blocks
        input_conv_kernel=[7, 7],
    ):  # kernel size of conv2d before dense blocks
        self.blocks = blocks
        self.growth_rate = growth_rate
        self.training = training
        self.dense_channels = dense_channels
        self.level = level
        self.dropout_rate = dropout_rate
        self.transition = transition
        self.input_conv_kernel = input_conv_kernel
        self.input_conv_stride = input_conv_stride
        self.input_conv_filters = input_conv_filters
        self.B_branch_from = B_option["branch_from"]
        self.B_growth_rate = B_option["growth_rate"]  # k
        self.B_level = B_option["level"]

    def bound(self, nin, nout, kernel):
        fin = nin * kernel[0] * kernel[1]
        fout = nout * kernel[0] * kernel[1]
        return np.sqrt(6.0 / (fin + fout))

    def dense_net(self, input_x):
        """
        Return dense_out = A, B

        Parameters:
        input_x: [batch, h, w, c=1]

        Returns:
        A_out
        B_out
        """
        #### before flowing into dense blocks ####
        x = input_x
        limit = self.bound(1, self.input_conv_filters, self.input_conv_kernel)
        x = tf.layers.conv2d(
            x,
            filters=self.input_conv_filters,
            strides=self.input_conv_stride,
            kernel_size=self.input_conv_kernel,
            padding="SAME",
            data_format="channels_last",
            use_bias=False,
            kernel_initializer=tf.random_uniform_initializer(
                -limit, limit, dtype=tf.float32
            ),
        )
        x = tf.layers.batch_normalization(
            x,
            training=self.training,
            momentum=0.9,
            scale=True,
            gamma_initializer=tf.random_uniform_initializer(
                -1.0 / math.sqrt(self.input_conv_filters),
                1.0 / math.sqrt(self.input_conv_filters),
                dtype=tf.float32,
            ),
            epsilon=0.0001,
        )
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(
            inputs=x, pool_size=[2, 2], strides=2, padding="SAME"
        )
        input_pre = x
        self.dense_channels += self.input_conv_filters
        dense_out = x
        #### flowing into dense blocks and transition_layer ####

        for i in range(self.blocks):
            for j in range(self.level):

                #### [1, 1] convolution part for bottleneck ####
                limit = self.bound(self.dense_channels, 4 * self.growth_rate, [1, 1])
                x = tf.layers.conv2d(
                    x,
                    filters=4 * self.growth_rate,
                    kernel_size=[1, 1],
                    strides=1,
                    padding="VALID",
                    data_format="channels_last",
                    use_bias=False,
                    kernel_initializer=tf.random_uniform_initializer(
                        -limit, limit, dtype=tf.float32
                    ),
                )
                x = tf.layers.batch_normalization(
                    inputs=x,
                    training=self.training,
                    momentum=0.9,
                    scale=True,
                    gamma_initializer=tf.random_uniform_initializer(
                        -1.0 / math.sqrt(4 * self.growth_rate),
                        1.0 / math.sqrt(4 * self.growth_rate),
                        dtype=tf.float32,
                    ),
                    epsilon=0.0001,
                )
                x = tf.nn.relu(x)
                x = tf.layers.dropout(
                    inputs=x, rate=self.dropout_rate, training=self.training
                )

                #### [3, 3] convolution part for regular convolve operation
                limit = self.bound(4 * self.growth_rate, self.growth_rate, [3, 3])
                x = tf.layers.conv2d(
                    x,
                    filters=self.growth_rate,
                    kernel_size=[3, 3],
                    strides=1,
                    padding="SAME",
                    data_format="channels_last",
                    use_bias=False,
                    kernel_initializer=tf.random_uniform_initializer(
                        -limit, limit, dtype=tf.float32
                    ),
                )
                x = tf.layers.batch_normalization(
                    inputs=x,
                    training=self.training,
                    momentum=0.9,
                    scale=True,
                    gamma_initializer=tf.random_uniform_initializer(
                        -1.0 / math.sqrt(self.growth_rate),
                        1.0 / math.sqrt(self.growth_rate),
                        dtype=tf.float32,
                    ),
                    epsilon=0.0001,
                )
                x = tf.nn.relu(x)
                x = tf.layers.dropout(
                    inputs=x, rate=self.dropout_rate, training=self.training
                )
                # print(f'x b4 {x.shape.as_list()}')
                dense_out = tf.concat([dense_out, x], axis=3)
                x = dense_out
                # print(f'x after {x.shape.as_list()}')
                #### calculate the filter number of dense block's output ####
                self.dense_channels += self.growth_rate

            if i < self.blocks - 1:
                compressed_channels = int(self.dense_channels * self.transition)

                #### new dense channels for new dense block ####
                self.dense_channels = compressed_channels
                limit = self.bound(self.dense_channels, compressed_channels, [1, 1])
                x = tf.layers.conv2d(
                    x,
                    filters=compressed_channels,
                    kernel_size=[1, 1],
                    strides=1,
                    padding="VALID",
                    data_format="channels_last",
                    use_bias=False,
                    kernel_initializer=tf.random_uniform_initializer(
                        -limit, limit, dtype=tf.float32
                    ),
                )
                x = tf.layers.batch_normalization(
                    x,
                    training=self.training,
                    momentum=0.9,
                    scale=True,
                    gamma_initializer=tf.random_uniform_initializer(
                        -1.0 / math.sqrt(self.dense_channels),
                        1.0 / math.sqrt(self.dense_channels),
                        dtype=tf.float32,
                    ),
                    epsilon=0.0001,
                )
                x = tf.nn.relu(x)
                x = tf.layers.dropout(
                    inputs=x, rate=self.dropout_rate, training=self.training
                )

            if i == self.B_branch_from:
                B_out = x

            if i < self.blocks - 1:
                x = tf.layers.average_pooling2d(
                    inputs=x, pool_size=[2, 2], strides=2, padding="SAME"
                )
                dense_out = x

            # print(f'x {x.shape.as_list()}')
            # print(f'dense {dense_out.shape.as_list()}')

        B = B_out
        for j in range(self.B_level):
            #### [1, 1] convolution part for bottleneck ####
            limit = self.bound(self.dense_channels, 4 * self.B_growth_rate, [1, 1])
            B = tf.layers.conv2d(
                B,
                filters=4 * self.B_growth_rate,
                kernel_size=[1, 1],
                strides=1,
                padding="VALID",
                data_format="channels_last",
                use_bias=False,
                kernel_initializer=tf.random_uniform_initializer(
                    -limit, limit, dtype=tf.float32
                ),
            )
            B = tf.layers.batch_normalization(
                inputs=B,
                training=self.training,
                momentum=0.9,
                scale=True,
                gamma_initializer=tf.random_uniform_initializer(
                    -1.0 / math.sqrt(4 * self.B_growth_rate),
                    1.0 / math.sqrt(4 * self.B_growth_rate),
                    dtype=tf.float32,
                ),
                epsilon=0.0001,
            )
            B = tf.nn.relu(B)
            B = tf.layers.dropout(
                inputs=B, rate=self.dropout_rate, training=self.training
            )

            #### [3, 3] convolution part for regular convolve operation
            limit = self.bound(4 * self.B_growth_rate, self.B_growth_rate, [3, 3])
            B = tf.layers.conv2d(
                B,
                filters=self.B_growth_rate,
                kernel_size=[3, 3],
                strides=1,
                padding="SAME",
                data_format="channels_last",
                use_bias=False,
                kernel_initializer=tf.random_uniform_initializer(
                    -limit, limit, dtype=tf.float32
                ),
            )
            B = tf.layers.batch_normalization(
                inputs=B,
                training=self.training,
                momentum=0.9,
                scale=True,
                gamma_initializer=tf.random_uniform_initializer(
                    -1.0 / math.sqrt(self.growth_rate),
                    1.0 / math.sqrt(self.growth_rate),
                    dtype=tf.float32,
                ),
                epsilon=0.0001,
            )
            B = tf.nn.relu(B)
            B = tf.layers.dropout(
                inputs=B, rate=self.dropout_rate, training=self.training
            )
            B_out = tf.concat([B_out, B], axis=3)
            B = B_out
        # A_out: [A]
        # B_out: [B]

        return dense_out, B_out

def main(args=None):

    # Data Loading
    # worddicts = load_dict(args.dictPath)
    dictLen = 111 #len(worddicts)
    # worddicts_r = [None] * len(worddicts)
    # for kk, vv in worddicts.items():
    #     worddicts_r[vv] = kk

    # train, train_uid_list = dataIterator(
    #     args.trainPklPath,
    #     args.trainCaptionPath,
    #     worddicts,
    #     batch_size=args.batch_size,
    #     batch_Imagesize=500000,
    #     maxlen=200,
    #     maxImagesize=500000,
    # )

    # valid, valid_uid_list = dataIterator(
    #     args.validPklPath,
    #     args.validCaptionPath,
    #     worddicts,
    #     batch_size=args.batch_size,
    #     batch_Imagesize=500000,
    #     maxlen=200,
    #     maxImagesize=500000,
    # )

    # print("train lenth is ", len(train))
    # print("valid lenth is ", len(valid))


    # Model
    # [bat, h, w, 1]
    x = tf.placeholder(tf.float32, shape=[None, None, None, 1])

    lr = tf.placeholder(tf.float32, shape=())

    if_trainning = tf.placeholder(tf.bool, shape=())

    B_option = {"branch_from": 1, "growth_rate": 24, "level": 8}  # D/2

    watcher_train = Watcher_train(
        blocks=3, level=16, growth_rate=24, training=if_trainning, B_option=B_option
    )

    # annotation [bat, H, W, C]
    # mask [bat, H, W]
    A_annotation, B_annotation = watcher_train.dense_net(x)

    annotation = tf.concat([A_annotation, B_annotation], axis=-1) # [bat, H, W, C_A + C_B]

    out = tf.layers.dense(
        inputs=annotation,
        units=dictLen
    ) # [bat, H, W, dictLen]

    out = tf.reduce_max(
        out,
        axis=(1,2)
    )

    # out = tf.nn.max_pool(
    #     value=out,
    #     ksize=(1,None, None, dictLen),
    #     strides=(1,1,1,1),
    #     padding="SAME"
    # ) # [bat, dictLen]

    # Training configuration
    vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    max_epoch = 200

    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    # model.compile(
    #     optimizer=tf.train.AdadeltaOptimizer(1.0),
    #     loss="binary_crossentropy",
    #     metrics=["accuracy"]
    # )
    return

    saver = tf.train.Saver()

    for epoch in range(max_epoch):
        for batch_x, batch_y in train:
            batch_x, _, batch_y, _ = prepare_data(batch_x, batch_y)
            model.train_on_batch(batch_x, batch_y)

if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("dictPath", type=str)
    # parser.add_argument("trainPklPath", type=str)
    # parser.add_argument("trainCaptionPath", type=str)
    # parser.add_argument("validPklPath", type=str)
    # parser.add_argument("validCaptionPath", type=str)
    # parser.add_argument("--logPath", type=str)
    # parser.add_argument("--batch_size", type=int, default=6)
    # parser.add_argument("--lr", type=float, default=1)
    # parser.add_argument("--savePath", type=str, default="./model/")
    # parser.add_argument("--saveName", type=str)
    # (args, unknown) = parser.parse_known_args()
    # print(f"Run with args {args}")
    # main(args)

