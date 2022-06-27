"""
This is wsgi.py!

For using gunicorn!
"""
from app import app

if __name__=="__main()__":
    app.run()