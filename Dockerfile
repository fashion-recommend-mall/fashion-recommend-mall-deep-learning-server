FROM python:3.10.0

COPY . /www
WORKDIR /www

RUN python3 -m pip install -U pip
RUN pip3 install -r requirements.txt

CMD uwsgi uwsgi.ini