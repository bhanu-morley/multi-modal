"""
admin.py - Gradio Admin Dashboard for Multi-Modal Emotion App.

Handles:
  - Secure admin login (using MongoDB + hashed passwords)
  - Viewing recent user records (text + image + predictions)
  - Displaying stored images from GridFS
"""

import os
from io import BytesIO
from PIL import Image
from pymongo import MongoClient
from gridfs import GridFS
from werkzeug.security import check_password_hash
from dotenv import load_dotenv
import gradio as gr

# ======================================================
# 1. Database Initialization
# ======================================================
# Load .env configuration for MongoDB credentials
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI missing in .env - required for admin panel")

# Create database client and file storage handler (GridFS)
client = MongoClient(MONGO_URI)
db = client["multi-modal"]
fs = GridFS(db)

# ======================================================
# 2. Admin Authentication Logic
# ======================================================
def login_admin(username: str, password: str):
    """
    Verify admin credentials using MongoDB 'admins' collection.
    Passwords are stored as werkzeug hashes.

    Args:
        username (str): Admin username
        password (str): Plaintext password input

    Returns:
        tuple: (bool success, str message)
    """
    admin = db.admins.find_one({"username": username})
    if not admin:
        return False, "Invalid username"

    if not check_password_hash(admin["password"], password):
        return False, "Incorrect password"

    return True, "Login successful"

# ======================================================
# 3. Record Loading Logic
# ======================================================

def load_records(page: int = 1):
    """
    Fetches recent emotion classification records and images from MongoDB.

    Args:
        page (int): Pagination index, default = 1

    Returns:
        tuple: (records_table, image_gallery)
    """
    limit = 10                     # number of records per page
    skip = (page - 1) * limit      # how many records to skip for pagination

    # Fetch latest records from records collection
    docs = list(db.records.find().sort("timestamp", -1).skip(skip).limit(limit))
    table, images = [], []

    for d in docs:
        # Build each row for the data table
        table.append([
            str(d.get("timestamp", "")),
            d.get("text_input", ""),
            f"{d.get('text_prediction', '')} - {d.get('text_confidence', '')}",
            f"{d.get('face_prediction', '')} - {d.get('face_confidence', '')}",
            f"{d.get('fused_emotion', '')} - {d.get('fused_confidence', '')}",
        ])

        # Attempt to fetch corresponding image from GridFS if available
        img = None
        if d.get("image_file_id"):
            try:
                file_data = fs.get(d["image_file_id"])
                img = Image.open(BytesIO(file_data.read()))
                img.thumbnail((200, 200))  # Resize for gallery preview
            except Exception as e:
                print(f"GridFS read error: {e}")

        if img:
            images.append(img)

    return table, images

# ======================================================
# 4. Gradio Interface (Admin UI)
# ======================================================

with gr.Blocks(title="Admin Dashboard") as app:
    # --- Login Section ---
    gr.Markdown("### 🔐 Admin Login Portal")

    auth_state = gr.State(False)  # Tracks login state (True after valid login)
    uname = gr.Textbox(label="Admin ID")
    pwd = gr.Textbox(label="Password", type="password")
    login_btn = gr.Button("Login")
    msg = gr.Markdown("")  # Login feedback message

    def handle_login(username, password):
        ok, text = login_admin(username, password)
        return text, ok

    login_btn.click(handle_login, [uname, pwd], [msg, auth_state])

    # Admin Dashboard (hidden until login)
    with gr.Group(visible=False) as panel:
        gr.Markdown("## Admin Dashboard — Recent Records")

        table = gr.Dataframe(
            headers=["Time", "Text", "Text Emotion", "Face Emotion", "Fusion"],
            interactive=False
        )
        gallery = gr.Gallery(columns=4, rows=3, height="450px")

        refresh_btn = gr.Button("Refresh Logs")
        refresh_btn.click(load_records, None, [table, gallery])

    # Toggle visibility of admin panel based on login success
    login_btn.click(lambda ok: gr.update(visible=ok), auth_state, panel)

# ======================================================
# 5. Launch Application
# ======================================================
# Launches on port 7861 for admin access
app.launch(server_port=7861)
