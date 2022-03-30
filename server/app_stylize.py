# 后端入口文件 python-flask

# 文件读取
import os
import base64
import re

from io import BytesIO
from PIL import Image

# flask
from flask import Flask, request
from flask_cors import CORS # 允许跨域访问

# model
from neural_style import get_stylize

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return 'style transfer server running'

@app.route('/stylize-with-data', methods=['GET', 'POST'])
def stylize_with_data():
    
    if request.method == 'POST':
        # 表单数据是str类型
        sessionId = request.form['id'] # str
        styleIds = request.form['styleIds'] # str: ,分割的id
        print("check styleIds: ", styleIds)
        modelId = int(request.form['modelId']) # int

        contentData = request.form['contentData']
        
        # content
        # content_path
        content_path = './images/content_upload_save/'+sessionId+'.png' # 文件类型是否有影响
        # 保存content图片（存储用户上传的图片，而后读取）
        image_data = re.sub('^data:image/.+;base64,', '', contentData)
        image_content = Image.open(BytesIO(base64.b64decode(image_data)))
        image_content.save(content_path)
        
        # style

        # model
        # todo
        # assert modelId == 0, "暂定只允许0"
        style_num = 32
        model_path = './pytorch_models/epoch_2_Wed_Mar_30_num32.model'

        # output
        output_path = './images/output_web_app/'+sessionId+'.png'
        
        # 调用模型
        # 可以正常获取数据，网络CORS会报错
        get_stylize(content_path, styleIds, style_num, model_path)

        # 删除指定路径的文件
        os.remove(content_path)
        
        # 返回数据
        with open(os.path.join(os.path.dirname(__file__), output_path), 'rb') as f:
            return u"data:image/png;base64," + base64.b64encode(f.read()).decode('ascii')

    return 'get to stylize-with-data'

if __name__ == "__main__":
    app.run(port=5001, debug=True)
