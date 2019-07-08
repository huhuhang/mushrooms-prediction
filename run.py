from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
@app.route('/')
def index():
    return 'Please use the POST method to get predictions.'


@app.route("/", methods=["POST"])  # 请求方法为 POST
def inference():

    df = pd.read_csv("mushrooms.csv")  # 读取数据
    X = pd.get_dummies(df.iloc[:, 1:])  # 读取特征并独热编码
    query_df = pd.DataFrame([request.json])  # 将 JSON 变为 DataFrame
    query = pd.get_dummies(query_df).reindex(
        columns=X.columns, fill_value=0)  # 将请求数据 DataFrame 处理成独热编码样式

    clf = joblib.load('mushrooms.pkl')  # 加载模型
    preds = clf.predict(query)  # 模型推理结果
    preds_proba = clf.predict_proba(query)[0].max()  # 样本推理概率

    if preds_proba >= 0.8:
        clf.partial_fit(query, preds)
        joblib.dump(clf, "mushrooms.pkl")  # 保存增量训练后模型
        status = True
    else:
        status = False

    # 返回推理结果
    return jsonify({"prediction": list(preds), "predict_proba": round(float(preds_proba), 3), "partial_fit": status})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
