from flask import Flask, send_from_directory, abort, jsonify
from flask_cors import CORS
from flask_restx import Api
import os
from config import PORT, BASEDATA_DIR, SHARED_DIR

frontend_dist_path = os.path.join(os.path.dirname(__file__), "dist")

app = Flask(__name__, static_folder=frontend_dist_path, static_url_path='')

# enable CORS for clients on other origins (optional)
try:
    CORS(app)
except Exception:
    pass


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(frontend_dist_path, path)):
        return send_from_directory(frontend_dist_path, path)
    else:
        return send_from_directory(frontend_dist_path, "index.html")


@app.route("/api/basedata/<path:filename>", methods=["GET"])
def get_basedata_file(filename):
    # basic path sanity checks
    if ".." in filename or filename.startswith("/"):
        abort(400)
    filepath = os.path.join(BASEDATA_DIR, filename)
    if not os.path.isfile(filepath):
        abort(404)
    return send_from_directory(BASEDATA_DIR, filename, mimetype="application/xml")

# Configure Flask-RESTX API
api = Api(
    app,
    version="1.0",
    title="Wearable REST API",
    description="REST API for Wearable services",
    doc="/api/docs",  # Swagger UI
)

# Register RESTX namespaces
from ms.restapi_directory import directory_ns
api.add_namespace(directory_ns)

from wde.interactivity.service_inter import interactivity_ns
from wde.usability.service_usb import usability_ns
from wde.wearability.service_wear import wearability_ns
from ms.db import initialize_db
from ms.restapi_setting import setting_ns

api.add_namespace(interactivity_ns)
api.add_namespace(usability_ns)
api.add_namespace(wearability_ns)
api.add_namespace(setting_ns)

initialize_db()


from ms.filecheck import FileHandler
from watchdog.observers import Observer
import time
import threading

def callback_file_change(event_type, src_path):
    print(f"File {event_type}: {src_path}")


def start_observer():
    event_handler = FileHandler(callback_file_change)
    observer = Observer()
    observer.schedule(event_handler, path=SHARED_DIR, recursive=True)
    observer.start()
    
    print(f"Started file observer on {SHARED_DIR}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    

if __name__ == "__main__":
    print(f"Server started on port {PORT}")
    
    observer_thread = threading.Thread(target=start_observer, daemon=True)
    observer_thread.start()
    
    app.run(host="0.0.0.0", port=PORT, debug=False)
