#       ┌─────────────────────┐                          ┌─────────────────┐      
#       │                     │                          │                 │
# ─────▶│ Transaction Record  ├──────2. Fraud Risk ─────▶│  Inference API  │◀────┐
#       │                     │◀────Prediction with ─────┤                 │     │
#       │                     │        Sub Graph         │                 │     │
#       └─────────────────────┘                          └─────────────────┘     │
#            │           ▲                                        │              │
#            │           │                                        │              │
#        0. Insert   1. Get New                              3.req: Node         │
#          Record    Record Sub                            Classification        │
#            │         Graph                                      │              │
#            ▼           │                                        │              │
# ┌──────────────────────┴─────────────────┐ ┌────────────────────┘      3.resp: │
# │┌──────────────────────────────────────┐│ │                          Predicted│
# ││   Graph of Historical Transactions   ││ │                             Risk  │
# │└──────────────────────────────────────┘│ │                                   │
# │                   .─.              .   │ │                                   │
# │                  (   )◀───────────( )  │ │                                   │
# │                   `─'              '   │ │      ┌──────────────────────┐     │
# │  .       .─.       ╲             ◁     │ │      │ GNN Model Λ          │     │
# │ ( )◀────(   )       ╲           ╱      │ │  ┌───┴─┐        ╱ ╲      ┌──┴──┐  │
# │  '       `─'         ╲       . ╱       │ │  ├─────┤       ╱   ╲     ├─────┤  │
# │  ╲       ◀            ╲     ( )        │ └─▶├─────┼─────▶▕     ─────├─────┤──┘
# │   ╲  .  ╱              ◁     '         │    ├─────┤       ╲   ╱     ├─────┤   
# │    ◀( )╱               .─.         .─. │    └───┬─┘        ╲ ╱      └──┬──┘   
# │      '                (   )◀──────(   )│        │           V          │      
# │                        `─'         `─' │        └──────────────────────┘      
# └────────────────────────────────────────┘

# Flask App for the Fraud Detection Backend
# RESTful API for New Record Ingestion and Inference
# SocketIO for broadcast new record to Dashboard

import os
import json
import gzip

import urllib.request as urllib2

import torch

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_socketio import emit

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

from .gnn import gnn_model, do_inference, get_subgraph, to_dgl

def create_instance():

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret!'
    app.config['TESTING'] = True
    app.config['ENVRIONMENT'] = 'development'
    CORS(app, supports_credentials=True)
    socketio = SocketIO(
        app, cors_allowed_origins=['http://10.1.1.127:8080'],
        logger=True, engineio_logger=True)


    @app.route("/api")
    def root():
        return jsonify({"status": "ok"})

    @socketio.on('my_event')
    def my_event(data):
        emit('my response', {'data': 'got it!'})
        print('received message: ' + data)

    @socketio.on('test', namespace="/")
    def test():  
        emit("my_event", broadcast=True)
        print("test recieved")

    @app.route("/api/add_review", methods=["POST"])
    def add_review():
        """
        curl -X POST 127.0.0.1:5000/api/add_review \
            -d '{"vertex_id": "2048"}' \
            -H 'Content-Type: application/json'
        """

        request_data = request.get_json()
        vertex_id = int(request_data.get("vertex_id", "2048"))
        with connection_pool.session_context("root", "nebula") as session:
            sub_graph = get_subgraph(session, vertex_id)
        g, node_id_map = to_dgl(sub_graph)
        pred = do_inference(
            graph=g,
            node_idx=node_id_map[vertex_id],
            model=model).tolist()
        # see below
        is_fraud = bool((pred[1] - pred[0]) > 2.0)
        # let's broadcast the new record to the dashboard
        emit("new_review", json.dumps({
            "vertex_id": vertex_id,
            "is_fraud": bool(is_fraud),
            "feat" : g.ndata["feat"][0].tolist()}),
            broadcast=True,
            namespace="/")
        return jsonify({"is_fraud": is_fraud})
        
        # We could see those fraud data comes with pred[1] - pred[0] > 2.0
        # In [369]: index_i = 0

        # In [370]: for r_ in result:
        #      ...:     if r_[0] < r_[1]:
        #      ...:         print(str(r_) + ": "+ str(hg_test.ndata['label'][index_i]))
        #      ...:     index_i += 1
        #      ...:
        # tensor([-0.4653,  2.0515]): tensor(1)
        # tensor([-0.3854,  1.3931]): tensor(1)
        # tensor([-1.1046,  2.5900]): tensor(1)
        # tensor([-0.7643,  1.8981]): tensor(1)
        # tensor([-0.9891,  1.9606]): tensor(1)
        # tensor([-0.8305,  2.1667]): tensor(1)
        # tensor([-0.5225,  1.8346]): tensor(1)
        # tensor([-0.5545,  1.8461]): tensor(1)
        # tensor([-0.7387,  1.7869]): tensor(1)
        # tensor([-0.8426,  1.9154]): tensor(1)
        # tensor([-1.1986,  1.8055]): tensor(1)
        # tensor([-0.9264,  2.2199]): tensor(1)
        # tensor([-1.2745,  2.5923]): tensor(1)
        # tensor([-0.6774,  1.9199]): tensor(1)
        # tensor([-0.8768,  1.4789]): tensor(1)
        # tensor([-0.5717,  0.7597]): tensor(0)
        # tensor([-0.2695,  1.6488]): tensor(0)
        # tensor([-1.1005,  1.9970]): tensor(1)
        # tensor([-1.1340,  2.2460]): tensor(1)
        # tensor([-0.9052,  2.2489]): tensor(1)
        # tensor([-0.1937,  1.3082]): tensor(1)
        # tensor([-0.1576,  1.4129]): tensor(1)
        # tensor([-0.8976,  1.9711]): tensor(1)
        # tensor([-0.9199,  2.1958]): tensor(1)
        # tensor([0.2016, 0.8248]): tensor(1)
        # tensor([-0.4452,  1.5325]): tensor(1)
        # tensor([-0.6960,  1.7163]): tensor(1)
        # tensor([0.6867, 0.7077]): tensor(0)
        # tensor([-0.7088,  1.5952]): tensor(1)
        # tensor([-0.9164,  2.2585]): tensor(1)
        # tensor([-0.7836,  2.2548]): tensor(1)
        # tensor([0.7041, 0.7049]): tensor(0)
        # tensor([-0.6718,  1.7835]): tensor(1)
        # tensor([-0.6657,  1.9698]): tensor(1)
        # tensor([-0.8179,  1.7891]): tensor(1)
        # tensor([-0.4910,  1.6841]): tensor(1)
        # tensor([-0.3798,  1.2956]): tensor(1)
        # tensor([-1.2844,  2.6029]): tensor(1)
        # tensor([-1.3074,  2.6277]): tensor(1)
        # tensor([-0.4255,  1.2594]): tensor(0)
        # tensor([-0.5220,  1.6514]): tensor(1)
        # tensor([-0.2769,  1.5949]): tensor(0)
        # tensor([0.6179, 1.1152]): tensor(0)
        # tensor([-0.6887,  1.7105]): tensor(1)
        # tensor([0.0102, 1.0130]): tensor(1)
        # tensor([-0.3933,  1.2951]): tensor(0)
        # tensor([-0.4631,  1.3833]): tensor(0)
        # tensor([-0.9345,  2.5175]): tensor(1)
        # tensor([-0.7374,  1.7976]): tensor(1)
        # tensor([-0.4642,  1.1329]): tensor(0)
        # tensor([-0.2998,  1.6415]): tensor(1)
        # tensor([0.0586, 1.7155]): tensor(0)
        # tensor([-0.8710,  2.5655]): tensor(1)
        # tensor([-0.1306,  1.4905]): tensor(0)
        # tensor([-0.3283,  1.2049]): tensor(1)
        # tensor([-0.6847,  1.8038]): tensor(1)
        # tensor([-0.6173,  1.7529]): tensor(1)
        # tensor([-1.0240,  2.4342]): tensor(1)
        # tensor([0.8416, 0.8935]): tensor(0)
        # tensor([-0.9524,  2.4270]): tensor(1)
        # tensor([-0.7680,  1.6052]): tensor(1)
        # tensor([-0.2216,  1.4630]): tensor(1)


    def parse_nebula_graphd_endpoint():
        ng_endpoints_str = os.environ.get(
            'NG_ENDPOINTS', '127.0.0.1:9669,').split(",")
        ng_endpoints = []
        for endpoint in ng_endpoints_str:
            if endpoint:
                parts = endpoint.split(":")  # we dont consider IPv6 now
                ng_endpoints.append((parts[0], int(parts[1])))
        return ng_endpoints


    ng_config = Config()
    ng_config.max_connection_pool_size = int(
        os.environ.get('NG_MAX_CONN_POOL_SIZE', 10))
    ng_endpoints = parse_nebula_graphd_endpoint()
    connection_pool = ConnectionPool()

    MODEL_LOCAL_PATH = "fraud_d.model"

    # load model from online gzip file
    DEFAULT_MODEL_URL = "https://github.com/wey-gu/NebulaGraph-Fraud-Detection-GNN/releases/download/v0.0.1/fraud_d.model.gz"

    def load_model(url):
        if not os.path.exists(MODEL_LOCAL_PATH):
            # Read the file inside the .gz archive located at url
            with urllib2.urlopen(url) as response:
                with gzip.GzipFile(fileobj=response) as uncompressed:
                    file_content = uncompressed.read()
            # write to file in binary mode 'wb'
            with open(MODEL_LOCAL_PATH, 'wb') as f:
                f.write(file_content)
        gnn_model.load_state_dict(torch.load(MODEL_LOCAL_PATH))
        return gnn_model


    connection_pool.init(ng_endpoints, ng_config)
    model = load_model(os.environ.get('NG_MODEL_URL', DEFAULT_MODEL_URL))

    return app
