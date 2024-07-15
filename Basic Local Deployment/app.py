import os
import time
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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global variables
pc = None
pinecone_index = None
index_name = 'question-answering'  # Use a fixed name

# Initialize SentenceTransformer
retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

# Define a Hugging Face model to use
HF_MODEL_NAME = "vblagoje/bart_lfqa"

def initialize_pinecone():
    global pc, pinecone_index, index_name
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Check if the index already exists
    if index_name not in pc.list_indexes().names():
        # Create a new index if it doesn't exist
        pc.create_index(
            name=index_name,
            dimension=768,  # This should match the dimension of your embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud='gcp',
                region='us-central1'  # This is the region for the gcp-starter environment
            )
        )
    
    # Connect to the index
    pinecone_index = pc.Index(index_name)
    
    print(f"Connected to Pinecone index: {index_name}")

# Call this function at the start of your application
initialize_pinecone()

def load_model():
    print(f"Loading model from Hugging Face: {HF_MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
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
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=150, min_length=40, do_sample=True, top_p=0.95, top_k=50)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def filter_answer(answer, question):
    # Remove answers that are too short or just repetitions of the question
    if len(answer) < 10 or answer.lower() in question.lower():
        return None
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    global pinecone_index
    
    video_url = request.json['video_url']
    try:
        video_id = video_url.split('v=')[1].split('&')[0]
    except IndexError:
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    try:
        # Clear existing vectors
        pinecone_index.delete(delete_all=True)
        
        # Process the video
        video_data, video_info = preprocess_video(video_id)
        
        # Convert to DataFrame
        df = pd.DataFrame(video_data)
        
        # Upsert to the Pinecone index
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
            "duration": video_info['duration'],
            "index_name": index_name
        })
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error processing video: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400

@app.route('/chat', methods=['POST'])
def chat():
    global pinecone_index
    
    user_message = request.json['message']
    
    try:
        # Query Pinecone to get relevant context
        query_embedding = retriever.encode([user_message]).tolist()
        query_response = pinecone_index.query(vector=query_embedding[0], top_k=3, include_metadata=True)
        
        if query_response['matches']:
            contexts = [match['metadata']['chunk_text'] for match in query_response['matches']]
            combined_context = " ".join(contexts)
            
            # Generate answer using the model
            answer = generate_answer(user_message, combined_context)
            
            # Filter the answer
            filtered_answer = filter_answer(answer, user_message)
            
            if filtered_answer:
                print(f"Generated answer: {filtered_answer}")
                return jsonify({"response": filtered_answer})
            else:
                print("No suitable answer generated")
                most_relevant_chunk = query_response['matches'][0]['metadata']['chunk_text']
                return jsonify({"response": f"I couldn't generate a specific answer, but here's the most relevant information I found: {most_relevant_chunk}"})
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