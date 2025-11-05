
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

PORT = int(os.getenv("PORT", 5000))
BASE_DIR = os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__))))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))


BASEDATA_DIR = os.path.join(BASE_DIR, "basedata")
SHARED_DIR = os.path.join(PARENT_DIR, "Patient")

print(f"BASE_DIR is set to: {BASE_DIR}")
print(f"PARENT_DIR is set to: {PARENT_DIR}")
print(f"BASEDATA_DIR is set to: {BASEDATA_DIR}")
print(f"SHARED_DIR is set to: {SHARED_DIR}")

os.makedirs(SHARED_DIR, exist_ok=True)