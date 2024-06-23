from flask import Flask, request, jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'  # Path to your service account JSON file
PARENT_FOLDER_ID = '1HY2wp0T1uxy07S0oXX4ZKULkbxWtg3UB'  # Your Google Drive folder ID

def authenticate():
    """Authenticate and return the Drive API service."""
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

def upload_file(file_name, file_content):
    """Upload a file to Google Drive."""
    service = authenticate()

    file_metadata = {
        'name': file_name,
        'parents': [PARENT_FOLDER_ID]
    }
    media = MediaIoBaseUpload(io.StringIO(file_content), mimetype='text/plain')

    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file

@app.route('/article', methods=['POST'])
def upload():
    """Handle the upload request from Google Apps Script."""
    try:
        data = request.get_json()
        file_name = data.get('fileName')
        file_content = data.get('fileContent')

        logging.info(f'Received request to upload file: {file_name}')

        if not file_name or not file_content:
            return jsonify({'success': False, 'error': 'Invalid input data'}), 400

        file = upload_file(file_name, file_content)
        logging.info(f'File uploaded successfully: {file_name} with ID: {file.get("id")}')

        return jsonify({'success': True, 'fileId': file.get('id')})
    except Exception as e:
        logging.error(f'Error uploading file: {str(e)}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def home():
    """Basic route for testing if server is running."""
    return "Server is running"

if __name__ == '__main__':
    app.run(port=5000, debug=True)
