## Arch and Flow

![GraphSAGE_FraudDetection](https://user-images.githubusercontent.com/1651790/182301784-21850dac-0d47-4dd5-b66f-a28b87fe9d4d.svg)

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