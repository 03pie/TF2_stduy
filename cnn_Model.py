#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (C) 2021 03pie, Inc. All Rights Reserved 
#
# @Time    : 2021/5/8 14:34
# @Author  : 03pie
# @Email   : 1139004179@qq.com
# @File    : cnn_Model.py
# @Software: PyCharm
# @Description :
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,MaxPool2D,GlobalAvgPool2D,Conv2D,BatchNormalization,Activation,Dropout

# BaseL_Model
class Base_Model(Model):
    def __init__(self):
        super(Base_Model, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = self.c1(inputs)
        inputs = self.b1(inputs)
        inputs = self.a1(inputs)
        inputs = self.p1(inputs)
        inputs = self.d1(inputs)

        inputs = self.flatten(inputs)
        inputs = self.f1(inputs)
        inputs = self.d2(inputs)
        outputs = self.f2(inputs)
        return outputs
# LeNet
class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()

        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='valid')
        # self.b1 = BatchNormalization()
        self.a1 = Activation('sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        # self.d1 = Dropout(0.0)

        self.c2 = Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid')
        # self.b2 = BatchNormalization()
        self.a2 = Activation('sigmoid')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        # self.d2 = Dropout()

        self.flatten = Flatten()
        self.d31 = Dense(120, activation='sigmoid')
        self.d32 = Dense(84, activation='sigmoid')
        self.d33 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = self.c1(inputs)
        inputs = self.a1(inputs)
        inputs = self.p1(inputs)

        inputs = self.c2(inputs)
        inputs = self.a2(inputs)
        inputs = self.p2(inputs)

        inputs = self.flatten(inputs)
        inputs = self.d31(inputs)
        inputs = self.d32(inputs)
        outputs = self.d33(inputs)

        return outputs
# AlexNet
class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.p5 = MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')

        self.flatten = Flatten()
        self.f61 = Dense(2048, activation='relu')
        self.d61 = Dropout(0.5)
        self.f62 = Dense(2048, activation='relu')
        self.d62 = Dropout(0.5)
        self.f63 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = self.c1(inputs)
        inputs = self.b1(inputs)
        inputs = self.a1(inputs)
        inputs = self.p1(inputs)

        inputs = self.c2(inputs)
        inputs = self.b2(inputs)
        inputs = self.a2(inputs)
        inputs = self.p2(inputs)

        inputs = self.c3(inputs)

        inputs = self.c4(inputs)

        inputs = self.c5(inputs)
        inputs = self.p5(inputs)

        inputs = self.flatten(inputs)
        inputs = self.f61(inputs)
        inputs = self.d61(inputs)
        inputs = self.f62(inputs)
        inputs = self.d62(inputs)
        outputs = self.f63(inputs)

        return outputs
# VGG16Net
class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')

        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')
        self.b4 = BatchNormalization()
        self.a4 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()
        self.a5 = Activation('relu')

        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = BatchNormalization()
        self.a6 = Activation('relu')

        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p7 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d7 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = BatchNormalization()
        self.a8 = Activation('relu')

        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = BatchNormalization()
        self.a9 = Activation('relu')

        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p10 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d10 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = BatchNormalization()
        self.a11 = Activation('relu')

        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = BatchNormalization()
        self.a12 = Activation('relu')

        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p13 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d13 = Dropout(0.2)

        self.flatten = Flatten()

        self.f14 = Dense(512, activation='relu')
        self.d14 = Dropout(0.2)

        self.f15 = Dense(512, activation='relu')
        self.d15 = Dropout(0.2)

        self.f16 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = self.c1(inputs)
        inputs = self.b1(inputs)
        inputs = self.a1(inputs)

        inputs = self.c2(inputs)
        inputs = self.b2(inputs)
        inputs = self.a2(inputs)
        inputs = self.p2(inputs)
        inputs = self.d2(inputs)

        inputs = self.c3(inputs)
        inputs = self.b3(inputs)
        inputs = self.a3(inputs)

        inputs = self.c4(inputs)
        inputs = self.b4(inputs)
        inputs = self.a4(inputs)
        inputs = self.p4(inputs)
        inputs = self.d4(inputs)

        inputs = self.c5(inputs)
        inputs = self.b5(inputs)
        inputs = self.a5(inputs)

        inputs = self.c6(inputs)
        inputs = self.b6(inputs)
        inputs = self.a6(inputs)

        inputs = self.c7(inputs)
        inputs = self.b7(inputs)
        inputs = self.a7(inputs)
        inputs = self.p7(inputs)
        inputs = self.d7(inputs)

        inputs = self.c8(inputs)
        inputs = self.b8(inputs)
        inputs = self.a8(inputs)

        inputs = self.c9(inputs)
        inputs = self.b9(inputs)
        inputs = self.a9(inputs)

        inputs = self.c10(inputs)
        inputs = self.b10(inputs)
        inputs = self.a10(inputs)
        inputs = self.p10(inputs)
        inputs = self.d10(inputs)

        inputs = self.c11(inputs)
        inputs = self.b11(inputs)
        inputs = self.a11(inputs)

        inputs = self.c12(inputs)
        inputs = self.b12(inputs)
        inputs = self.a12(inputs)

        inputs = self.c13(inputs)
        inputs = self.b13(inputs)
        inputs = self.a13(inputs)
        inputs = self.p13(inputs)
        inputs = self.d13(inputs)

        inputs = self.flatten(inputs)

        inputs = self.f14(inputs)
        inputs = self.d14(inputs)

        inputs = self.f15(inputs)
        inputs = self.d15(inputs)

        outputs = self.f16(inputs)

        return outputs
# Inception
class ConvBNRelu(Model):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
       super(ConvBNRelu, self).__init__()
       self.model = tf.keras.models.Sequential([
           Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding),
           BatchNormalization(),
           Activation('relu')])

    def call(self, inputs, training=None, mask=None):
        outputs = self.model(inputs, training=False)
        return outputs

class InceptionBLK(Model):
    def __init__(self, filters, strides=1):
        super(InceptionBLK, self).__init__()
        self.filters = filters
        self.strides = strides
        self.c1 = ConvBNRelu(filters, kernel_size=1, strides=strides)
        self.c2_1 = ConvBNRelu(filters, kernel_size=1, strides=strides)
        self.c2_2 = ConvBNRelu(filters, kernel_size=3, strides=1)
        self.c3_1 = ConvBNRelu(filters, kernel_size=1, strides=strides)
        self.c3_2 = ConvBNRelu(filters, kernel_size=5, strides=1)
        self.c4_1 = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')
        self.c4_2 = ConvBNRelu(filters, kernel_size=1, strides=strides)

    def call(self, inputs, training=None, mask=None):
        outputs_x1 = self.c1(inputs)
        outputs_x2_1 = self.c2_1(inputs)
        outputs_x2_2 = self.c2_2(outputs_x2_1)
        outputs_x3_1 = self.c3_1(inputs)
        outputs_x3_2 = self.c3_2(outputs_x3_1)
        outputs_x4_1 = self.c4_1(inputs)
        outputs_x4_2 = self.c4_2(outputs_x4_1)

        outputs = tf.concat([outputs_x1, outputs_x2_2, outputs_x3_2, outputs_x4_2], axis=3)
        return outputs
class Inception10(Model):
    def __init__(self, num_blocks=2, num_classes=10, init_filters=16, **kwargs):
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_filters
        self.out_channels = init_filters
        self.num_blocks = num_blocks
        self.init_filters = init_filters
        self.c1 = ConvBNRelu(init_filters)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBLK(self.out_channels, strides=2)
                else:
                    block = InceptionBLK(self.out_channels, strides=1)
                self.blocks.add(block)
            self.out_channels *= 2
        self.p1 = GlobalAvgPool2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = self.c1(inputs)
        inputs = self.blocks(inputs)
        inputs = self.p1(inputs)
        outputs = self.f1(inputs)

        return outputs

# ResNet18
class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

    def __init__(self, block_list=[2, 2, 2, 2], initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y



