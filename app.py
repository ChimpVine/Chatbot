from flask import Flask, redirect, url_for, request, jsonify, render_template_string
from flask_cors import CORS
from model import process_query

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# In-memory storage for queries and responses
query_response_store = []


@app.route('/')
def index():
    # Redirect the root URL to /process_query
    return redirect(url_for('process_query_endpoint'))


@app.route('/process_query', methods=['GET', 'POST'])
def process_query_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({'error': 'Please provide a query'}), 400

        try:
            response = process_query(query)
            query_response_store.append({'query': query, 'response': response})
            # Debug information
            print(f"Stored query-response pair: {query_response_store[-1]}")
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    elif request.method == 'GET':
        # Handle GET request, display stored queries and responses
        return render_template_string('''
            <html>
            <head><title>Query Responses</title></head>
            <body>
                <h1>Stored Queries and Responses</h1>
                <table border="1">
                    <tr><th>Query</th><th>Response</th></tr>
                    {% for item in query_response_store %}
                    <tr><td>{{ item.query }}</td><td>{{ item.response }}</td></tr>
                    {% endfor %}
                </table>
            </body>
            </html>
        ''', query_response_store=query_response_store)


if __name__ == '__main__':
    # Define the port to run the Flask application on
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
