## How To Run

First, install and run the backend with `flask run`.

```bash
python3 -m pip install -r requirements.txt

export NG_ENDPOINTS="10.1.1.168:9669";
export FLASK_ENV=development;
export FLASK_APP=wsgi;

python3 -m flask run --reload
```

Then, set up an Nginx to enable CORS, the configuration is under `../nginx.conf`
