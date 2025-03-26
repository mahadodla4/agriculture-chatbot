from email import message
from urllib import response
from flask import Flask, render_template, request, jsonify, request_finished
from testing_model_trans import generate_response, generate_embeddings, load_model_and_context
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

context_file_path= r'C:\Users\sirij\agriculture-chatbot\context.txt'

context, model, context_embeddings = load_model_and_context(context_file_path)

def load_context_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"Error reading context file: {e}")
        return ""

@app.get("/")
def index_get():
    return render_template("chatbot.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # context_file = load_context_from_txt(context)
    response = generate_response(text,context,model,context_embeddings)
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