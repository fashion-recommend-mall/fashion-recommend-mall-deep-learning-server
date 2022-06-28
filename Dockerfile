FROM python:3.10.0

COPY . /www
WORKDIR /www

RUN python3 -m pip install -U pip
RUN pip3 install -r resource/requirements.txt

CMD gunicorn -c gunicorn.config.py