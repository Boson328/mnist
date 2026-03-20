import datetime
import keras
from keras import layers, models
from typing import cast, Any
import tensorflow as tf

# 1. データのロード（自動でダウンロードされます）
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 正規化（0-1の範囲に）
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

def residual_block_x2(x, filters, strides=1):
    original_x = x;
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=strides)(x)
    x = layers.BatchNormalization()(x) # 正規化
    x = layers.Activation("mish")(x) # 活性化関数

    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x) # 正規化

    original_x = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides)(original_x)

    x = layers.Add()([x, original_x])
    x = layers.Activation("mish")(x)
    return x


def conv1(x):
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("mish")(x)
    return x

def conv2_x(x):
    # x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    x = residual_block_x2(x, 64)
    x = residual_block_x2(x, 64)
    return x

def conv3_x(x):
    x = residual_block_x2(x, 128, 2)
    x = residual_block_x2(x, 128)
    x = residual_block_x2(x, 128)
    return x

def conv4_x(x):
    x = residual_block_x2(x, 256, 2)
    x = residual_block_x2(x, 256)
    x = residual_block_x2(x, 256)
    return x

def conv5_x(x):
    x = residual_block_x2(x, 512, 2)
    x = residual_block_x2(x, 512)
    x = residual_block_x2(x, 512)
    return x




# 2. モデル構築（カラー写真用に少し深くします）
def build_cifar_model():
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        # # ±10%（約36度）の範囲でランダムに回転
        keras.layers.RandomRotation(0.1),
        # # 上下左右に±10%の範囲でランダムにズラす
        keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        # # ±10%の範囲でランダムにズーム
        keras.layers.RandomZoom(0.08),
    ])

    inputs = layers.Input(shape=(32, 32, 3)) # RGBの3チャンネル！
    x = data_augmentation(inputs)

    # conv1の部分
    x = conv1(x)

    x = conv2_x(x)
    x = conv3_x(x)

    x = conv4_x(x)
    # 8, 8, 64
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("mish")(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)

    return model


model = build_cifar_model()

# 3. コンパイル
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. 学習（RTX 3060ならサクサク動くはずです）
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model_name = f"./cifar10/model_{now_str}.keras"

model.save(model_name)

