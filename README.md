tmux -u new -s fastapi
server: PYTHONIOENCODING=utf-8 gunicorn server:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 1800
client: client.py