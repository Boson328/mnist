import datetime
import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# 固定シード（ガチャの再現性をなくすなら削除してもOK）
#tf.random.set_seed(42)
#np.random.seed(42)

# --- 1. データの準備 ---
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test = (x_test.astype("float32") / 255.0)[..., None]

# --- 2. モデル構築用関数の定義 ---
def build_new_model():
    data_augmentation = keras.Sequential([
        # ±10%（約36度）の範囲でランダムに回転
        keras.layers.RandomRotation(0.06),
        # 上下左右に±10%の範囲でランダムにズラす
        keras.layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
        # ±10%の範囲でランダムにズーム
        keras.layers.RandomZoom(0.09),
    ])

    model = keras.models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        data_augmentation, # あなたが作った水増し層
        
        layers.Conv2D(32, 3, padding='same', activation='mish'), # 16→32へ
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, 3, padding='same', activation='mish'), # 128→64へ（高速化）
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='mish'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',                   # 効率よく学習を進めるためのアルゴリズム
        loss='sparse_categorical_crossentropy', # 正解との「ズレ」を計算する方法
        metrics=['accuracy']                # 「正解率」を画面に表示する
    )
    return model

class GachaCheck(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.is_failed_gacha = False

    def on_epoch_end(self, epoch, logs=None):
        # logs が None の場合のガード + model が None の場合のガード
        if logs is not None and self.model is not None and epoch == 2:
            # dict.get() ではなく明示的に取得し、型を確定させる
            acc = logs.get('accuracy', 0.0)
            if acc < 0.80:
                print(f"\n[!] 精度不足 ({acc:.4f})。初期値ハズレと判断。")
                self.model.stop_training = True
                self.is_failed_gacha = True

# --- 4. メイン学習ループ ---
MAX_TRIALS = 15 # 2時間ならこれくらい余裕を持って設定
for trial in range(1, MAX_TRIALS + 1):
    # 毎回違うシード値を使って「運」を変える
    seed = int(datetime.datetime.now().timestamp()) + trial
    tf.random.set_seed(seed)
    np.random.seed(seed)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"model_trial{trial}_{now_str}.keras"
    
    print(f"\n{'#'*60}")
    print(f" 試行 {trial}/{MAX_TRIALS} (Seed: {seed})")
    print(f" 保存先: {model_name}")
    print(f"{'#'*60}\n")

    model = build_new_model()
    gacha_callback = GachaCheck()

    callbacks = [
        # より細かく学習率を下げる (factor=0.5)
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        # 99.6%付近の戦いは長いので、patienceを長めに設定
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    try:
        history = model.fit(
            x_train, y_train,
            epochs=100,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            batch_size=128
        )
        
        # ガチャに成功（＝途中で止まらずに完走）し、かつ高精度なら終了
        if not gacha_callback.is_failed_gacha:
            final_val_acc = max(history.history['val_accuracy'])
            if final_val_acc > 0.996: # 目標精度
                print(f"\n[🌟] 目標精度 {final_val_acc:.4f} を達成！完了します。")
                break
            else:
                print(f"\n[?] 完走しましたが精度が目標({final_val_acc:.4f})に届きませんでした。次へ。")
            
    except KeyboardInterrupt:
        print("\n[!] 中断。")
        break

print(f"\n[🏁] 2時間の実験工程が完了しました。")
