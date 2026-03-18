import glob
import keras
import numpy as np
from typing import cast, List, Tuple

# 1. データの準備（評価用）
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = (x_test.astype("float32") / 255.0)[..., None]

# 2. 保存されたモデルファイルを全て取得
model_files = glob.glob("*.keras")
model_files.sort()

results: List[Tuple[str, float]] = []

print(f"\n{'='*60}")
print(f"{'Model Name':<35} | {'Test Accuracy':<15}")
print(f"{'-'*60}")

for f in model_files:
    try:
        # load_modelの結果を keras.Model 型として明示的にキャスト
        loaded_obj = keras.models.load_model(f)
        if loaded_obj is not None:
            model = cast(keras.Model, loaded_obj)
            # これで .evaluate が認識される
            loss_acc = model.evaluate(x_test, y_test, verbose="0")
            
            # evaluateは [loss, accuracy] のリストを返す
            acc = float(loss_acc[1]) 
            results.append((f, acc))
            print(f"{f:<35} | {acc:.5f}")
            
    except Exception as e:
        print(f"{f:<35} | Error: {e}")

print(f"{'='*60}")

# 3. 最高のモデルを特定
if results:
    best_model = max(results, key=lambda x: x[1])
    print(f"\n[🏆] 最高のモデル: {best_model[0]} (Accuracy: {best_model[1]:.5f})")
