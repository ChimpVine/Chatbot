from flask import Flask, render_template
import json

app = Flask(__name__, template_folder='template')

# Load FAQs from JSON file
with open('faqs.json', 'r') as file:
    faqs = json.load(file)

@app.route('/', methods=['GET'])
def render_home():
    return render_template('faq.html', faqs=faqs)

if __name__ == '__main__':
    app.run(debug=True)
