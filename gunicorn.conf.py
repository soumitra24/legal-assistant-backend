import os

bind = f"0.0.0.0:{os.getenv('PORT', 8000)}"
workers = int(os.getenv('WORKERS', 4))
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"