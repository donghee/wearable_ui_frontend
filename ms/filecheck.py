
from watchdog.events import FileSystemEventHandler

class FileHandler(FileSystemEventHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def on_modified(self, event):
        if not event.is_directory:
            self.callback("modified", event.src_path)
            
    def on_created(self, event):
        if not event.is_directory:
            self.callback("created", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self.callback("deleted", event.src_path)