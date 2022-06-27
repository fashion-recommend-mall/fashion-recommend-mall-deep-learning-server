from datetime import datetime

from flask import request, Blueprint, Response, json
from flask import current_app
import urllib.request
import ssl

from src.service import style_categorize_service


ssl._create_default_https_context = ssl._create_unverified_context
app = Blueprint('app', __name__, url_prefix='/')


@app.route('/upload', methods=['GET'])
def categorize():

    ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr) 

    img_path = request.args.get("img_path")

    current_app.logger.info(f'\n \
     REQUEST\n \
     Method : {request.method} \n \
     IP : {ip_address} \n \
     Img path : {img_path}')

    file_path = 'static/uploads' + f'/{datetime.now()}.jpg'

    urllib.request.urlretrieve(img_path, file_path)

    result = style_categorize_service(file_path)

    current_app.logger.info(f'\n \
     RESPONSE\n \
     Code : {200} \n \
     Data : {result} ')
    
    return Response(json.dumps(result), status=200, mimetype='application/json')