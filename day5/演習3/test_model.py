# 推論精度を評価するシンプルなスクリプト
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

# モデル読み込み
with open("day5/演習1/model/model.pkl", "rb") as f:
    model = pickle.load(f)

# テストデータ読み込み
X_test = pd.read_csv("day5/演習1/data/X_test.csv")
y_test = pd.read_csv("day5/演習1/data/y_test.csv")

# 推論と精度計算
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
