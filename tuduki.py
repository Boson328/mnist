import keras

# 1. 前回保存したモデルを読み込む
model = keras.models.load_model('conv_6.keras')

# 2. そのまま追加で10エポック回す
model.fit(train_generator, epochs=10, validation_data=(x_test, y_test))

# 3. 強化されたモデルを上書き保存
model.save('conv_2_improved.keras')
