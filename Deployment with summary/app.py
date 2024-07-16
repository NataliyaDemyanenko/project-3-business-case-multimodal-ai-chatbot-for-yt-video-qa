import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Define the index name
index_name = 'question-answering'

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # This should match the dimension of your embeddings
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'  # Change this to your preferred region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Test the index connection
try:
    stats = index.describe_index_stats()
    print(f"Successfully connected to Pinecone index. Total vectors: {stats['total_vector_count']}")
except Exception as e:
    print(f"Error connecting to Pinecone index: {str(e)}")
    raise

# Global variable to store the index
pinecone_index = index

# Initialize SentenceTransformer
retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

# Define the path to your local model (if you have one)
LOCAL_MODEL_PATH = "./fine_tuned_model"

# Define a Hugging Face model to use (you can change this to any model you prefer)
HF_MODEL_NAME = "distilbert-base-cased-distilled-squad"

def load_model():
    # Check if we have a local model
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            print(f"Attempting to load model from {LOCAL_MODEL_PATH}")
            model = AutoModelForQuestionAnswering.from_pretrained(LOCAL_MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
            print("Successfully loaded local model")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading local model: {str(e)}")
    
    # If no local model or loading failed, use the Hugging Face model
    print(f"Loading model from Hugging Face: {HF_MODEL_NAME}")
    model = AutoModelForQuestionAnswering.from_pretrained(HF_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    print("Successfully loaded Hugging Face model")
    return model, tokenizer

# Load the model
model, tokenizer = load_model()

def preprocess_video(video_id):
    print(f"Starting to preprocess video: {video_id}")
    ydl_opts = {'skip_download': True}
    video_data = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Extracting video info...")
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        print("Video info extracted successfully")

        print("Downloading transcript...")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print("Transcript downloaded successfully")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)

        combined_text = ""
        combined_start_time = 0
        combined_duration = 0

        print("Processing transcript...")
        for i, entry in enumerate(transcript):
            combined_text += entry['text'] + " "
            if i == 0:
                combined_start_time = entry['start']
            combined_duration += entry['duration']

            if combined_duration >= 150 or i == len(transcript) - 1:
                chunk_texts = text_splitter.split_text(combined_text)
                for j, text in enumerate(chunk_texts):
                    chunk_id = f"{video_id}_{i}_{j}"
                    video_data.append({
                        'chunk_id': chunk_id,
                        'video_id': video_id,
                        'title': info['title'],
                        'channel': info['uploader'],
                        'description': info['description'],
                        'duration': info['duration'],
                        'start_time': combined_start_time,
                        'end_time': combined_start_time + combined_duration,
                        'source': f"https://www.youtube.com/watch?v={video_id}&t={int(combined_start_time)}",
                        'chunk_text': text
                    })
                combined_text = ""
                combined_start_time = 0
                combined_duration = 0
        print("Transcript processing completed")

        return video_data, info
    except Exception as e:
        print(f"Error in preprocess_video: {str(e)}")
        raise

def generate_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    if answer_start >= answer_end:
        return "I couldn't find a specific answer to that question in the given context."
    
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    # Remove [CLS], [SEP], and special tokens
    answer = answer.replace("[CLS]", "").replace("[SEP]", "").strip()
    
    if not answer:
        return "I couldn't find a specific answer to that question in the given context."
    
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    video_url = request.json['video_url']
    try:
        video_id = video_url.split('v=')[1].split('&')[0]  # This will handle URLs with additional parameters
    except IndexError:
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    try:
        # Process the video
        video_data, video_info = preprocess_video(video_id)
        
        # Convert to DataFrame
        df = pd.DataFrame(video_data)
        
        # Upsert to Pinecone
        batch_size = 100
        for i in tqdm(range(0, len(df), batch_size)):
            i_end = min(i + batch_size, len(df))
            batch = df.iloc[i:i_end]
            emb = retriever.encode(batch['chunk_text'].tolist()).tolist()
            
            upsert_data = [
                (str(row['chunk_id']), vec, row.to_dict())
                for vec, (_, row) in zip(emb, batch.iterrows())
            ]
            
            try:
                pinecone_index.upsert(vectors=upsert_data)
            except Exception as e:
                print(f"Error during upsert: {str(e)}")
                raise
        
        return jsonify({
            "message": "Video processed successfully",
            "video_id": video_id,
            "title": video_info['title'],
            "channel": video_info['uploader'],
            "description": video_info['description'],
            "duration": video_info['duration']
        })
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error processing video: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    try:
        # Query Pinecone to get relevant context
        query_embedding = retriever.encode([user_message]).tolist()
        query_response = pinecone_index.query(vector=query_embedding[0], top_k=1, include_metadata=True)
        
        if query_response['matches']:
            context = query_response['matches'][0]['metadata']['chunk_text']
            print(f"Found context: {context[:100]}...")  # Print first 100 characters of context
            
            # Generate answer using the model
            answer = generate_answer(user_message, context)
            
            if answer.strip() == "[CLS]" or not answer.strip():
                print("Model returned empty or [CLS] answer")
                return jsonify({"response": "I'm sorry, I couldn't generate a relevant answer based on the available information."})
            
            print(f"Generated answer: {answer}")
            return jsonify({"response": answer})
        else:
            print("No matches found in Pinecone index")
            return jsonify({"response": "I'm sorry, I couldn't find relevant information to answer your question."})
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in chat: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400

if __name__ == '__main__':
    app.run(debug=True)