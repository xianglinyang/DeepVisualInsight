from flask import request, Flask, jsonify, make_response
from flask_cors import CORS, cross_origin

import os, sys
import numpy as np
import base64
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from demo_mlp_mnist import demo_mlp_mnist
# flask for API server
app = Flask(__name__)
cors = CORS(app, origins="http://macbookpro:6006", supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/visualize', methods=["POST"])
@cross_origin()
def visualize():
    res = request.get_json()
    X = []
    Y = []

    for data in res["sampled_data"]:
        vector = []
        for dimension in data["vector"]:
            vector.append(data["vector"][dimension])
        X.append(vector)
        Y.append(data["metadata"]["label"])

    X = X[:500]
    Y = Y[:500]
    X = np.array(X)
    Y = np.array(Y)
    result = demo_mlp_mnist(X,Y).tolist()
    with open("cls_view.png", "rb") as f:
        bg = base64.b64encode(f.read()).decode('utf-8')
    return make_response(jsonify({'result': result, 'bg':bg}), 200)

@app.route('/', methods=["POST", "GET"])
def index():
    return 'Index Page'


@app.route('/hello')
def hello_world():
    return 'Hello World!'


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":

    app.run(host="192.168.1.115")