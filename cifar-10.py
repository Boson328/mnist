import datetime
import keras
from keras import layers
from typing import cast, Any

# 1. データのロード（自動でダウンロードされます）
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 正規化（0-1の範囲に）
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

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

    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)), # RGBの3チャンネル！
        data_augmentation,
        
        # 第1ブロック
        layers.Conv2D(32, (3, 3), padding='same', activation='mish'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='mish'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # 第2ブロック（チャンネル数を増やして特徴を捉える）
        layers.Conv2D(128, (3, 3), padding='same', activation='mish'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='mish'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128, activation='mish'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])
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

