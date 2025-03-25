from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

context=r'C:\Users\sirij\agriculture-chatbot\context.txt'

# Load models (these should be loaded once at startup)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2") 
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_response(text, context):
    try:
        answer = answer_question_with_patterns(text, context)
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