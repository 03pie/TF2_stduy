#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# Copyright (C) 2021 03pie, Inc. All Rights Reserved 
#
# @Time    : 2021/5/8 14:22
# @Author  : 03pie
# @Email   : 1139004179@qq.com
# @File    : 1_6.py
# @Software: PyCharm
# @Description :
import os
import tensorflow as tf
import cnn_Model
import matplotlib.pyplot as plt

# 设置初始化


model_ = 'ResNet18'
lr = 0.001
epochs = 20
batch_size = 32
# model = cnn_Model.Base_Model()
# model = cnn_Model.LeNet()
# model = cnn_Model.AlexNet()
# model = cnn_Model.VGG16()
# model = cnn_Model.Inception10()
model = cnn_Model.ResNet18()
#------------------------------------------------------------------------------------------------------------------------------------

# 路径设置

root_path = './'
plt_name = root_path + 'loss/' + model_ + '.png'
checkpoint_path = root_path + 'checkpoint/' + model_ + '/'
ckpt = model_ + '.ckpt'

# 准备训练集和验证集

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# 模型编译

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy'])

# 模型读取

checkpoint_save_path = root_path + checkpoint_path + ckpt
if os.path.exists(checkpoint_save_path + '.index'):
    print('---------------------load the model------------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_best_only=True,
    save_weights_only=True)


# 模型训练

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    validation_freq=1,
    callbacks=[cp_callback])

# 模型信息打印

model.summary()

# acc,log绘制

print('All epochs has trained, log is plotting...')
train_acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(1, 2, 1)
plt.title('acc')
plt.plot(train_acc, label='Train_acc')
plt.plot(val_acc, label='Val_acc')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('loss')
plt.plot(train_loss, label='Train_loss')
plt.plot(val_loss, label='Val_loss')
plt.legend()

plt.savefig(plt_name)
