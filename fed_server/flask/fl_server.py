from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# 模拟全局序列数据 (seq_len=20, n_features=3)
GLOBAL_SEQ = np.random.rand(100, 20, 3).tolist()  # 100 个样本

# 当前分发索引
current_idx = 0

@app.route('/seq_input', methods=['GET'])
def get_seq_input():
    global current_idx
    t = request.args.get('t', default=0, type=int)

    # 循环分发
    sample = GLOBAL_SEQ[current_idx % len(GLOBAL_SEQ)]
    current_idx += 1

    return jsonify(sample)

@app.route('/upload', methods=['POST'])
def upload_client_data():
    data = request.json
    # data 可以包含 client_id, features, labels, model_params 等
    print("Received from client:", data.keys())
    return jsonify({'status':'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
