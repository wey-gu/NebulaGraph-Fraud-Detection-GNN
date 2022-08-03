## How To Run

First, install and run the backend with `flask run`.

```bash
python3 -m pip install -r requirements.txt

export NG_ENDPOINTS="127.0.0.1:9669";
export FLASK_ENV=development;
export FLASK_APP=wsgi;

python3 -m flask run --reload --host=0.0.0.0
```

Then, set up an Nginx to enable CORS, the configuration is under `../nginx.conf`

