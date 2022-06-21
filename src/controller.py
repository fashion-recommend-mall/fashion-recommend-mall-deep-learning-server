from flask import request, Blueprint, Response, json
import urllib.request
from src.service import style_categorize_service
from datetime import datetime

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

app = Blueprint('app', __name__, url_prefix='/')

@app.route('/upload', methods=['GET'])
def categorize():

    img_path = request.args.get("img_path")

    file_path = 'static/uploads' + f'/{datetime.now()}.jpg'

    urllib.request.urlretrieve(img_path, file_path)

    result = style_categorize_service(file_path)
    
    return Response(json.dumps(result), status=200, mimetype='application/json')