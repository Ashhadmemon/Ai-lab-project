import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import app

application = app

# Vercel discovers a WSGI application from `app` or `application`.
app = application
