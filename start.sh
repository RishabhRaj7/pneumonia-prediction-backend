gunicorn --chdir app -b 0.0.0.0:$PORT pneumonia_backend:app
