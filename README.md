## Arch and Flow

![GraphSAGE_FraudDetection](https://user-images.githubusercontent.com/1651790/182623863-de5c8ba6-5107-4707-8122-d2130085d5ac.svg)

### Model Training

Check https://github.com/wey-gu/NebulaGraph-Fraud-Detection-GNN/tree/main/notebooks/Train_GraphSAGE.ipynb for details.

- Input: Graph of Historical Yelp Reviews
- Output: a GraphSAGE Node Classification Model, could be inductive

```asciiarmor
                     ┌──────────────────────────────────────────────┐                     
                     │   ┌──────────────────────────────────────┐   │                     
                     │   │     Graph of Historical Reviews      │   │                     
                     │   └──────────────────────────────────────┘   │                     
                     │                      .─.              .      │                     
                     │                     (   )◀───────────( )     │                     
                     │                      `─'              '      │                     
                     │     .       .─.       ╲             ◁        │                     
                     │    ( )◀────(   )       ╲        .  ╱         │                     
                     │     '       `─'         ╲      ( )╱          │                     
                     │     ╲       ◀            ╲      '            │                     
                     │      ╲  .  ╱              ◁                  │                     
                     │       ◀( )╱               .─.         .─.    │                     
                     │         '                (   )◀──────(   )   │                     
                     │                           `─'         `─'    │                     
                     │                                              │                     
                     └──────────────────────────────────────────────┘                     
                                             ┃   (Nebula-DGL: NebulaLoader)                                         
                                             ▼                                            
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ┌────┐           ┌ ─ ─ ─ ─ ─ ─ ─ ─               ┌ ─ ─ ─ ─ ─ ─ ─ ─                     │
│ │GNN │                            │                               │                    │
│ └────┘           │                               │                                     │
│                                   │                               │                    │
│                  │           ◀                   │           ◀                         │
│                      .   .  ╱.─.  │                  .   .  ╱.─.  │                    │
│                  │  ( )◀───╱(   )                │  ( )◀───╱(   )                      │
│ .       .─.          '╱  '   `─'  │  ┌────────┐      '╱  '   `─'  │        .       .─. │
│( )◀────(   )     │   .       .─.     │ ReLU  ╱│  │   .       .─.          ( )◀────(   )│
│ '       `─'         ( )◀────(   ) │  │      ╱ │     ( )◀────(   ) │        '       `─' │
│ ╲       ◀   ══▶  │   '       `─'     │     ╱  │  │   '       `─'   ... ══▶ ╲       ◀   │
│  ╲  .  ╱             ╲       ◀    │  │─────   │      ╲       ◀    │         ╲  .  ╱    │
│   ◀( )╱          │    ╲  .  ╱        └────────┘  │    ╲  .  ╱                ◀( )╱     │
│     '                  ◀( )╱      │                    ◀( )╱      │            '       │
│                  │       '                       │       '                             │
│                      .       .─.  │                  .       .─.  │                    │
│                  │  ( )◀────(   )                │  ( )◀────(   )                      │
│                      '       `─'  │                  '       `─'  │                    │
│                  │   ╲       ◀                   │   ╲       ◀                         │
│                       ╲  .  ╱     │                   ╲  .  ╱     │                    │
│                  │     ◀( )╱                     │     ◀( )╱                           │
│                          '        │                      '        │                    │
│                  └ ─ ─ ─ ─ ─ ─ ─ ─               └ ─ ─ ─ ─ ─ ─ ─ ─                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            ┃                                             
                                            ▼                                             
                          ┌──────────────────────────────────┐                            
                          │                 Λ                │                            
                      ┌───┴─┐  GNN Model   ╱ ╲            ┌──┴──┐                         
                      ├─────┤             ╱   ╲           ├─────┤                         
                      ├─────┼────────────▶    ───────────▶├─────┤                         
                      ├─────┤             ╲   ╱           ├─────┤                         
                      └───┬─┘              ╲ ╱            └──┬──┘                         
                          │                 V                │                            
                          └──────────────────────────────────┘      
```

### Online Fraud Inference System

Check https://github.com/wey-gu/NebulaGraph-Fraud-Detection-GNN/tree/main/notebooks/Inference_API.ipynb for details.

#### Backend
- Input: a new review
- Output: is_fraud prediction
- Flow:
  0. A review will be inserted to NebulaGraph
  1. A SubGraph Query will be called
  2. SubGraph will be sent to Inference API
  3. Inference API will predict its `is_fraud` label on the trained model

```asciiarmor
      ┌─────────────────────┐                          ┌─────────────────┐      
      │                     │                          │                 │
─────▶│ Transaction Record  ├──────2. Fraud Risk ─────▶│  Inference API  │◀────┐
      │                     │◀────Prediction with ─────┤                 │     │
      │                     │        Sub Graph.        │                 │     │
      └─────────────────────┘                          └─────────────────┘     │
           │           ▲                                        │              │
           │           │                                        │              │
       0. Insert   1. Get New                              3.req: Node         │
         Record.   Record Sub                            Classification.       │
           │         Graph.                                     │              │
           ▼           │                                        │              │
┌──────────────────────┴─────────────────┐ ┌────────────────────┘      3.resp: │
│┌──────────────────────────────────────┐│ │                          Predicted│
││   Graph of Historical Transactions   ││ │                             Risk. │
│└──────────────────────────────────────┘│ │                                   │
│                   .─.              .   │ │                                   │
│                  (   )◀───────────( )  │ │                                   │
│                   `─'              '   │ │      ┌──────────────────────┐     │
│  .       .─.       ╲             ◁     │ │      │ GNN Model Λ          │     │
│ ( )◀────(   )       ╲           ╱      │ │  ┌───┴─┐        ╱ ╲      ┌──┴──┐  │
│  '       `─'         ╲       . ╱       │ │  ├─────┤       ╱   ╲     ├─────┤  │
│  ╲       ◀            ╲     ( )        │ └─▶├─────┼─────▶▕     ─────├─────┤──┘
│   ╲  .  ╱              ◁     '         │    ├─────┤       ╲   ╱     ├─────┤   
│    ◀( )╱               .─.         .─. │    └───┬─┘        ╲ ╱      └──┬──┘   
│      '                (   )◀──────(   )│        │           V          │      
│                        `─'         `─' │        └──────────────────────┘      
└────────────────────────────────────────┘                                      
```

#### Frontend

As the review request being sent to Graph Database and Inference API, when fraud predict is responded to the Inference API caller, in parallel, the result will be broadcast to Real Time Fraud Monitor Dashboards, too.

The dashbard are tables subscribing to the flow of reviews sending in, and when some of the records are highlighted with hi risk in fraud, corresponding party will be notified and inovlved for follow-up actions.

Demo Video 👉🏻:
https://user-images.githubusercontent.com/1651790/182651965-d489a218-36a6-40c9-9fab-ba288e8d959a.mov

```asciiarmor
         ┌────────────────────────────────────────────────────────────────────┐         
         │   ┌──────────────────────────────────────────────────────────┐     │         
         │   │        Real-Time Online Fraud Monitor Web Service        │     │         
         │   └──────────────────────────────────────────────────────────┘     │         
         │                                                                    │         
         │   ┌────┬────┬──────┬────┬────┬────┬────┬────┬────┬────┬──────┐     │         
         │   │    │    │      │    │    │    │    │    │    │    │  OK  │     │         
         │   ├────┼────┼──────┼────┼────┼────┼────┼────┼────┼────┼──────┤     │         
         │   │    │    │      │    │    │    │    │    │    │    │  OK  │     │         
         │   ├────┼────┼──────┼────┼────┼────┼────┼────┼────┼────┼──────┤     │         
         │   │    │    │      │    │    │    │    │    │    │    │ NOK  │     │         
         │   └────┴────┴──────┴────┴────┴────┴────┴────┴────┴────┴──────┘     │         
         └─────────────────────────────────▲──────────────────────────────────┘         
                                           ┃                                            
┌───────────────────────┐                  ┃                                            
│  New Review/Requests  │                  ┃                                            
│Generated Continuously │                  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓               
└───────────────────────┘                                               ┃               
            │                                                           ┃               
            │                                                           ┃               
            │ ┌─────────────────────┐                          ┌────────┻────────┐      
            │ │                     │                          │                 │      
            │ │                     │                          │                 │      
            └▶│ Transaction Record  ├──────2. Fraud Risk ─────▶│  Inference API  │◀────┐
              │                     │◀────Prediction with ─────┤                 │     │
              │                     │        Sub Graph         │                 │     │
              └─────────────────────┘                          └─────────────────┘     │
...
```


## Graph Model and Data Set

We will leverage Yelp-Fraud dataset comes from [Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters](https://paperswithcode.com/paper/enhancing-graph-neural-network-based-fraud).

There will be one type of node and three types of edges:

- Node: review on restaurant, hotel. With Label and Feature Properties:
  - `is_fraud` to be the label
  - 32 features being feature-engineered 
- Edge: in 3-typed between the nodes. Without Properties:
  - R-U-R: share same reviewer, named `shares_user_with`
  - R-S-R: share same rate for same object, named `shares_restaurant_rating_with`
  - R-T-R: share same review submitting month for same object, named `shares_restaurant_in_one_month_with`

Before the project, I made the playground to ingest the Yelp Data Graph into NebulaGraph, see more from https://github.com/wey-gu/nebulagraph-yelp-frauddetection.

### Playground Setup with Data Ingestion

You could quickly run the following lines to make it ready:

```bash
# Deploy NebulaGraph for Playground
curl -fsSL nebula-up.siwei.io/install.sh | bash

# Clone the data downloader repo
git clone https://github.com/wey-gu/nebulagraph-yelp-frauddetection && cd nebulagraph-yelp-frauddetection

# Install requirement, then download the data ready for NebulaGraph
python3 -m pip install -r requirements.txt
python3 data_download.py

# Import it to NebulaGraph
docker run --rm -ti \
 --network=nebula-net \
 -v ${PWD}/yelp_nebulagraph_importer.yaml:/root/importer.yaml \
 -v ${PWD}/data:/root \
 vesoft/nebula-importer:v3.1.0 \
 --config /root/importer.yaml
```

Then refer to https://github.com/wey-gu/NebulaGraph-Fraud-Detection-GNN/tree/main/notebooks/ for Training Model and the Fraud Web Service itself.



### Playground of Real-Time Fraud Monitor

Get your machine's IP (not the 127.0.0.1), say it's `10.0.0.5`.

```bash
export MY_IP="10.0.0.5"
```

Run Backend:

```bash
git clone https://github.com/wey-gu/NebulaGraph-Fraud-Detection-GNN.git
cd NebulaGraph-Fraud-Detection-GNN/src

# ADD MY_IP into CORS & Frontend file, nginx.conf
sed -i "s/nebula-demo.siwei.io/$MY_IP/g" fraudd_backend/fraudd/__init__.py
sed -i "s/nebula-demo.siwei.io/$MY_IP/g" fraudd_frontend/src/components/Table.vue
sed -i "s/nebula-demo.siwei.io/$MY_IP/g" nginx.conf

# install dep of backend
python3 -m pip install -r requirements.txt

export NG_ENDPOINTS="127.0.0.1:9669";
export FLASK_ENV=development;
export FLASK_APP=wsgi;

# run backend
cd fraudd_backend

python3 -m flask run --reload --host=0.0.0.0
```

```bash
# verify
$ curl localhost:5000/api
{
  "status": "ok"
}
```

From another terminal, build frontend:

```bash
cd NebulaGraph-Fraud-Detection-GNN/src
cd fraudd_frontend

# sudo apt install npm

npm install
npm run build
```

From another terminal, run Nginx:

```bash
cd NebulaGraph-Fraud-Detection-GNN/src
docker-compose up -d
```

```bash
# end-to-end verify backend
curl -X POST localhost:15000/api/add_review \
        -d '{"vertex_id": "2049"}' \
        -H 'Content-Type: application/json'
```

```bash
# return value
{
  "is_fraud": false
}
```

From web browser 👉🏻 http://10.0.0.5:8080/

> You could check my demo: http://nebula-demo.siwei.io:8080

