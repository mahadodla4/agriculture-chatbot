from email import message
from urllib import response
from flask import Flask, render_template, request, jsonify, request_finished
from chat import get_response
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

@app.get("/")
def index_get():
    return render_template("chatbot.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    
    response = get_response(text)
    message={"answer":response}
    return jsonify(message)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    message = data['message']
    target_lang = data['target_lang']  # In this case, Telugu (te)

    # Translate message to the target language (Telugu)
    translated = translator.translate(message, dest=target_lang)
    response = {"translated_text": translated.text}
    return jsonify(response)

if __name__=="__main__":
    app.run(debug=True)