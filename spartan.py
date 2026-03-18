import keras
from typing import cast, Any

# データの準備
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test = (x_test.astype("float32") / 255.0)[..., None]

# 1. 既存モデルの読み込み
loaded_model = keras.models.load_model("best.keras")
if loaded_model is None:
    raise ValueError("モデルの読み込みに失敗しました。パスを確認してください。")
model = cast(keras.Model, loaded_model)

# ---------------------------------------------------------
# 【新機能】停滞したら最高地点からやり直すカスタムコールバック
# ---------------------------------------------------------
class RestoreAndContinue(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, patience=5):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.patience = patience  # 何エポック更新がなければ戻すか
        self.best_acc = 0.0
        self.wait = 0

    def on_train_begin(self, logs=None):
        # 最初の最高精度を現在のモデルから取得（あるいは0からスタート）
        self.best_acc = 0.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get("val_accuracy")
        if current_acc is None:
            return

        if current_acc > self.best_acc:
            self.best_acc = current_acc
            self.wait = 0
            # 最高記録をファイルに保存
            self.model.save(self.checkpoint_path)
            print(f"\n[✨] 最高精度更新: {current_acc:.5f} - 重みを保存しました")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n[🔄] {self.patience}エポック停滞中... 最高地点 ({self.best_acc:.5f}) の重みを復元して続行します")
                self.model.load_weights(self.checkpoint_path)
                self.wait = 0

# ---------------------------------------------------------

# 2. 追加訓練用の設定
# 学習率を 5e-6 よりもさらに細かく 2e-6 に設定（微調整用）
opt = keras.optimizers.Adam(learning_rate=2e-6)

# 3. コンパイル
model.compile(
    optimizer=cast(Any, opt),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# 4. コールバックの設定
# EarlyStopping を外し、自作の RestoreAndContinue を入れます
checkpoint_file = "model_refined_best.keras"
callbacks = [
    RestoreAndContinue(checkpoint_path=checkpoint_file, patience=8) # 8回更新できなければ戻す
]

# 5. 学習実行
# 途中で止まらないように epochs を多めに設定
model.fit(
    x_train, 
    y_train, 
    validation_data=(x_test, y_test), 
    epochs=100, 
    batch_size=256, 
    callbacks=callbacks,
    verbose=1
)
