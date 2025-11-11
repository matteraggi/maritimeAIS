from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os, pickle

# === CONFIGURAZIONE ===
LOCAL_FOLDER = "preprocessed"
DRIVE_FOLDER_NAME = "preprocessed"   # cartella di destinazione su Drive
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# === AUTENTICAZIONE GOOGLE ===
creds = None
if os.path.exists("token.pickle"):
    with open("token.pickle", "rb") as token:
        creds = pickle.load(token)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
    with open("token.pickle", "wb") as token:
        pickle.dump(creds, token)

drive = build("drive", "v3", credentials=creds)

# === CREA CARTELLA SE NON ESISTE ===
query = f"name='{DRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'"
res = drive.files().list(q=query, fields="files(id)").execute()
if res["files"]:
    folder_id = res["files"][0]["id"]
else:
    file_metadata = {"name": DRIVE_FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"}
    folder = drive.files().create(body=file_metadata, fields="id").execute()
    folder_id = folder.get("id")

# === UPLOAD DEI FILE ===
for filename in os.listdir(LOCAL_FOLDER):
    local_path = os.path.join(LOCAL_FOLDER, filename)
    if os.path.isfile(local_path):
        print(f"Caricamento: {filename}")
        media = MediaFileUpload(local_path, resumable=True)
        file_metadata = {"name": filename, "parents": [folder_id]}
        drive.files().create(body=file_metadata, media_body=media, fields="id").execute()

print(f"✅ Tutti i file caricati nella cartella '{DRIVE_FOLDER_NAME}' su Google Drive.")
