from flask import Flask, request, jsonify
import json

app = Flask(__name__)


@app.route('/role', methods=['POST'])
def roleinfo():
    request_data = json.loads(request.data)
    print(request_data)
    # print(request_data)
    ##解析json  ，获取参数，调用你的模型里的功能
    ###业务逻辑，可以调用你训练完的模型的功能

    # ret = classfier(image)

    # 把结果以json的方式返回
    return jsonify(request_data)


@app.route('/userinfo', methods=['GET'])
def userinfo():
    ##解析json  ，获取参数，调用你的模型里的功能或者方法（函数）

    ###业务逻辑，可以调用你训练完的模型的功能

    # ret = classfier(image)

    # 把结果以json的方式返回
    return jsonify({'text': "zhangsan"})

@app.route('/m',methods=['GET'])
def m():
    return jsonify({'id':'ok'})

if (__name__ == "__main__"):
    app.run(host='127.0.0.1', port=5000)





