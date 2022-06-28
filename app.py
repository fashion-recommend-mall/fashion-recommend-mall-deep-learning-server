"""
This is app.py!
"""
from flask import Flask
from flask_cors import CORS
from logging.config import dictConfig

import src.controller as controller


dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] [%(levelname)s] FROM [%(module)s] %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)

CORS(app)

app.register_blueprint(controller.app)

if __name__ == '__main__':    
    app.run(host='0.0.0.0', port=3000)