<div align="center">
  <h1>Mushrooms Prediction</h1>
</div>

<br />

## Usage

Run `python run.py` locally to start the Flask web service, or use the Heroku API:

```python
import requests
import pandas as pd

df = pd.read_csv("mushrooms.csv")  # 读取原数据集
sample_data = df.sample(1).to_json(orient='records')  # 从原数据中随机取 1 条用于测试推理，并处理成 JSON 类型
requests.post(url="https://mushrooms-prediction.herokuapp.com", json=sample_data).content  # 建立 POST 请求，并发送数据请求
```