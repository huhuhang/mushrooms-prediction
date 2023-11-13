<div align="center">
  <h1>🍄 Mushrooms Prediction</h1>
</div>

<br />

## Usage

Run `python run.py` locally to start the Flask web service, or use the Heroku API:

```python
import json
import requests
import pandas as pd

df = pd.read_csv("mushrooms_test.csv")  # 读取测试数据集
sample_data = df.sample(1).to_json()  # 从原数据中随机取 1 条用于测试推理，并转换成 JSON 样式
sample_json = json.loads(sample_data)  # 将 Pandas 转换的 JSON 样式数据处理成 JSON 类型
requests.post(url="https://mushrooms-prediction.herokuapp.com/", json=sample_json).content  # 建立 POST 请求，并发送数据请求
```
