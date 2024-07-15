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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-base")

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

def get_most_relevant_context(question, contexts, top_n=3):
    texts = [question] + contexts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    relevant_contexts = [contexts[i] for i in top_indices if cosine_similarities[i] > 0.2]  # Increased threshold
    return relevant_contexts

def generate_answer(question, context):
    input_text = f"Based on the following context, provide a concise and relevant answer to the question. If the context doesn't contain relevant information, say 'I don't have enough information to answer that question based on the video content.'\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            max_length=150,  # Reduced max_length for more concise answers
            min_length=20, 
            do_sample=True, 
            top_p=0.95, 
            top_k=50, 
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i+chunk_size]))
    return chunks

def create_summary(text):
    if not text:
        return "No text available to summarize."
    
    chunks = chunk_text(text)
    summaries = []
    
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {str(e)}")
            continue
    
    if not summaries:
        return "Unable to generate a summary."
    
    combined_summary = ' '.join(summaries)
    
    if len(combined_summary.split()) > 200:
        try:
            final_summary = summarizer(combined_summary, max_length=200, min_length=100, do_sample=False)[0]['summary_text']
        except Exception as e:
            print(f"Error in final summarization: {str(e)}")
            final_summary = combined_summary[:200] + "..."
    else:
        final_summary = combined_summary
    
    return final_summary
    
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
        
        # Store complete text
        complete_text = " ".join(df['chunk_text'])
        
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
        
        # Store complete text as a separate vector
        complete_text_embedding = retriever.encode([complete_text]).tolist()[0]
        pinecone_index.upsert(vectors=[("complete_text", complete_text_embedding, {"chunk_text": complete_text})])
        
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
    
    user_message = request.json['message'].lower().strip()
    
    try:
        summary_requests = [
            "summarize the video", "what is the video about", "what is the topic",
            "give me a summary", "what's the main idea", "what's the video's content",
            "provide an overview", "what are the key points"
        ]
        
        if any(request in user_message for request in summary_requests):
            return summarize_video()
        
        query_embedding = retriever.encode([user_message]).tolist()
        query_response = pinecone_index.query(vector=query_embedding[0], top_k=5, include_metadata=True)
        
        if query_response['matches']:
            contexts = [match['metadata']['chunk_text'] for match in query_response['matches']]
            most_relevant_contexts = get_most_relevant_context(user_message, contexts)
            
            if not most_relevant_contexts:
                return jsonify({"response": "I don't have enough information to answer that question based on the video content."})
            
            combined_context = " ".join(most_relevant_contexts)
            
            answer = generate_answer(user_message, combined_context)
            
            return jsonify({
                "response": answer,
                "context": combined_context[:500] + "..." if len(combined_context) > 500 else combined_context
            })
        else:
            return jsonify({"response": "I don't have enough information to answer that question based on the video content."})
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in chat: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400

def summarize_video():
    global pinecone_index
    
    try:
        # Retrieve all vectors from Pinecone
        query_response = pinecone_index.query(
            vector=[0]*768,  # Dummy vector
            top_k=10000,  # Retrieve all vectors
            include_metadata=True
        )
        
        if query_response['matches']:
            # Extract all text chunks
            all_text = " ".join([match['metadata']['chunk_text'] for match in query_response['matches'] if 'chunk_text' in match['metadata']])
            
            if not all_text:
                return jsonify({"response": "I don't have enough information to summarize the video content."})
            
            # Generate summary
            try:
                summary = create_summary(all_text)
                return jsonify({"response": f"Here's a summary of the video: {summary}"})
            except Exception as e:
                print(f"Error in create_summary: {str(e)}")
                return jsonify({"response": "I don't have enough information to summarize the video content."})
        else:
            return jsonify({"response": "I don't have enough information to summarize the video content."})
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in summarize_video: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400

if __name__ == '__main__':
    app.run(debug=True)