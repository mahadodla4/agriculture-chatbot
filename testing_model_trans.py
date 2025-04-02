from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

context=r'C:\Users\sirij\agriculture-chatbot\context.txt'

# Load models (these should be loaded once at startup)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2") 
model = SentenceTransformer('all-MiniLM-L6-v2')


def load_model_and_context(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            context = f.read()
        context_embeddings = generate_embeddings(context)
        return context, model, context_embeddings
    except FileNotFoundError:
        return "", None, None
    except Exception as e:
        print(f"Error reading context file: {e}")
        return "", None, None


# Generate embeddings for the context
def generate_embeddings(text):
    # Generate sentence embeddings for the context
    return model.encode([text])

# Find the most similar context and answer the question
def generate_response(text, context, model, context_embeddings):
    try:
        if text.lower() in ["hi","hello"]:
            return "Hello! How can I help you today?"
        elif text.lower() in ["bye"]:
            return "Goodbye! Have a great day!"
        elif text.lower() in ["thanks","thank you","thankyou"]:
            return "You're welcome!"
        elif text.lower() in ["how are you"]:
            return "I'm doing great! How can I help you today?"
        elif text.lower() in ["who are you","what is your name","what is your name?"]:
            return "I'm an AI agri chatbot! How can I help you today?"
        elif text.lower() in ["who created you","who developed you","who created you?","who developed you?"]: 
            return "I was developed by Mahalakshmi Dodla"
        else:
            question_embedding = model.encode([text])
            similarity = cosine_similarity(question_embedding, context_embeddings)
            
            # Get the most relevant context piece based on cosine similarity
            most_similar_context = context
            
            answer = answer_question(text, most_similar_context)
            return answer['answer']
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def answer_question_with_patterns(question, context):
    try:
        context_dict = json.loads(context.replace("'", "\""))
        patterns = context_dict.get("pattern", [])
        responses = context_dict.get("response", [])

        if patterns and responses:
            question_embedding = model.encode([question])[0]
            pattern_embeddings = model.encode(patterns)
            similarity_scores = cosine_similarity([question_embedding], pattern_embeddings)[0]
            most_similar_index = similarity_scores.argmax()
            return {"answer": responses[most_similar_index], "score": similarity_scores[most_similar_index]}
        else:
            return answer_question(question, context)
    except json.JSONDecodeError:
        return answer_question(question, context)
  
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result