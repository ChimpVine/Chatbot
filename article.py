from flask import Flask, request, jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'  # Path to your service account JSON file
PARENT_FOLDER_ID = '1HY2wp0T1uxy07S0oXX4ZKULkbxWtg3UB'  # Replace with your folder ID

def authenticate():
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

def upload_file(file_name, file_content):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': file_name,
        'parents': [PARENT_FOLDER_ID]
    }
    media = MediaIoBaseUpload(io.StringIO(file_content), mimetype='text/plain')

    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file

@app.route('/article', methods=['POST', 'OPTIONS'])
def upload():
    app.logger.debug(f"Received {request.method} request at /article")
    if request.method == 'OPTIONS':
        app.logger.debug("Handling preflight OPTIONS request")
        return jsonify({'success': True}), 200
    
    try:
        data = request.get_json()
        file_name = data['fileName']
        file_content = data['fileContent']

        file = upload_file(file_name, file_content)
        return jsonify({'success': True, 'fileId': file.get('id')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
