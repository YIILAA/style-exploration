from flask import Flask, request, jsonify
from flask_cors import CORS # 做跨域的准备
from flask import session # 追踪客户端会话

app = Flask(__name__) # 创建实例
app.secret_key = "super_secret_key" # ???

CORS(app) # 支持跨域访问

model = TextClassifier.load_from_file('models/best-model.pt') # 模型加载




@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)