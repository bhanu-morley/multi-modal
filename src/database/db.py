"""
db.py - Handles MongoDB connection and record storage for the app.
Uses environment variables from .env for secure configuration.
"""

from pymongo import MongoClient
from dotenv import load_dotenv
import os

# ====================================
# 1. Load Environment Variables
# ====================================
# Loads all key-value pairs from a .env file into system environment variables.
# This is safer than hardcoding credentials or URLs in the script.
load_dotenv()


# ====================================
# 2. Configuration Settings
# ====================================
# Get MongoDB URI (required) and database name (optional, default = "multi-modal")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB", "multi-modal")


# If MONGO_URI is missing, raise an error early —-don’t fail silently.
if not MONGO_URI:
    raise ValueError("MONGO_URI is missing! Add it to your .env file")


# ====================================
# 3. Connect to MongoDB
# ====================================
# Create a MongoDB client connection using the provided URI.
# This establishes a reusable connection pool for the whole app.
client = MongoClient(MONGO_URI)


# Select (or create if missing) the target database by name.
db = client[DB_NAME]


# ====================================
# 4. Helper Function to Save Data
# ====================================
def save_record(record: dict) -> bool:
    """
    Inserts a single document into the 'records' collection.
    Returns True if successful, False if an error occurs.

    Parameters:
        record (dict): The document to insert into MongoDB.

    Example:
        save_record({"user": "Bhanu", "emotion": "happy", "timestamp": "2025-11-10"})
    """
    try:
        db.records.insert_one(record)  # insert_one automatically creates the collection if it doesn't exist
        return True
    except Exception as e:
        print(f"MongoDB Error: {e}")
        return False
