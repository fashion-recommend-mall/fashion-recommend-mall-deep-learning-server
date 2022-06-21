from flask import Flask
from flask_cors import CORS
import src.controller as controller

app = Flask(__name__)
CORS(app)

app.register_blueprint(controller.app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)