import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D, Dense, Flatten
from tensorflow.keras.layers import Dropout, SpatialDropout3D, Activation, BatchNormalization, Input, concatenate
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 配置GPU使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存按需分配
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# 是否减少数据量
reduce_data = False  # 是否减少数据量
trial = '_trail_1'  # 试验标识
use_bias = False  # 是否使用偏置
data_type = 'COM'  # 数据类型
target = "3dir"  # 目标方向

# 提取各数据集
x_train = np.load('train_samples.npy')
y_train = np.load('train_porosities.npy')

x_test = np.load('test_samples.npy')
y_test = np.load('test_porosities.npy')

print("数据已加载")

# 调整数据形状以适应3D卷积
x_train = np.expand_dims(x_train, axis=4)
x_test = np.expand_dims(x_test, axis=4)

# CNN架构参数设置
batch_size = 32  # 批量大小
epochs = 35  # 训练轮数
activation = 'relu'  # 激活函数
loss_function = 'mean_absolute_percentage_error'  # 损失函数
optimizer = 'adam'  # 优化器


# 定义Inception模块
def inception_module(layer_in, f1, f2, strides):
    # 7x7卷积
    conv7 = layer_in
    conv7 = Conv3D(f1, 7, strides=strides, padding='same', use_bias=use_bias)(conv7)
    conv7 = Activation(activation)(conv7)
    conv7 = Conv3D(f1, 7, strides=strides, padding='same', use_bias=use_bias)(conv7)
    conv7 = Activation(activation)(conv7)

    # 15x15卷积
    conv15 = layer_in
    conv15 = Conv3D(f2, 15, strides=strides, padding='same', use_bias=use_bias)(conv15)
    conv15 = Activation(activation)(conv15)
    conv15 = Conv3D(f2, 15, strides=strides, padding='same', use_bias=use_bias)(conv15)
    conv15 = Activation(activation)(conv15)

    # 拼接两个分支
    layer_out = concatenate([conv7, conv15], axis=-1)
    return layer_out


# 定义模型
def build_model(input_shape):
    input_image = Input(shape=input_shape)

    # 添加Inception模块
    layer = inception_module(input_image, 16, 16, strides=2)
    layer = BatchNormalization()(layer)
    layer = Conv3D(16, 2, strides=2, padding='same', use_bias=use_bias)(layer)

    # 添加卷积块
    layer = Conv3D(32, 5, strides=1, padding='same', use_bias=use_bias)(layer)
    layer = Activation(activation)(layer)
    layer = Conv3D(32, 5, strides=1, padding='same', use_bias=use_bias)(layer)
    layer = Activation(activation)(layer)
    layer = BatchNormalization()(layer)
    layer = SpatialDropout3D(0.1)(layer)
    layer = Conv3D(32, 2, strides=2, padding='same', use_bias=use_bias)(layer)

    # 全连接层部分
    layer = Flatten()(layer)
    layer = Dense(128, use_bias=use_bias, activation=activation)(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(64, use_bias=use_bias, activation=activation)(layer)
    layer = Dense(1)(layer)

    model = Model(inputs=input_image, outputs=layer)
    return model


# 设置输入形状
input_shape = (100, 100, 100, 1)

# 创建模型
model = build_model(input_shape)
model.compile(optimizer=optimizer, loss=loss_function)

# 打印模型摘要
print(model.summary())

# 设置保存路径
save_name = f'{data_type}_{len(y_train)}_{optimizer}_{loss_function}_{epochs}_{target}_{activation}{trial}'
os.makedirs(save_name, exist_ok=True)  # 创建保存文件夹

# 定义回调函数
callbacks = [
    ModelCheckpoint(f'{save_name}/best_model.h5', save_best_only=True, monitor='val_loss', mode='min'),
    CSVLogger(f'{save_name}/training.log'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
]

# 训练模型
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,  # 使用20%的数据作为验证集
    shuffle=True,
    callbacks=callbacks
)

# 保存模型摘要到文本文件
with open(f'{save_name}/model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# 使用 model.save 保存整个模型
model.save(f'{save_name}/my_model.keras')

# 保存训练历史
np.save(f'{save_name}/history.npy', history.history)

# 绘制训练历史图
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{save_name}/loss_graph.png', dpi=300)
plt.clf()

# 加载模型
loaded_model = load_model(f'{save_name}/my_model.keras')

# 使用测试集进行预测
y_pred = loaded_model.predict(x_test)

np.save(f'{save_name}/y_pred.npy', y_pred)

# 计算R²
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2}")

# 计算R
r, _ = pearsonr(y_test.flatten(), y_pred.flatten())
print(f"R: {r}")

# 保存R和R²到文件
with open(f'{save_name}/metrics.txt', 'w') as f:
    f.write(f"R: {r}\n")
    f.write(f"R²: {r2}\n")