import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm import tqdm
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('question-answering')

# Initialize SentenceTransformer
retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

# Load the fine-tuned model
base_model = AutoModelForQuestionAnswering.from_pretrained("./fine_tuned_model")
peft_config = PeftConfig.from_pretrained("./fine_tuned_model")
model = PeftModel.from_pretrained(base_model, "./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

def preprocess_video(video_id):
    ydl_opts = {'skip_download': True}
    video_data = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)

    combined_text = ""
    combined_start_time = 0
    combined_duration = 0

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

    return video_data, info

def generate_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    video_url = request.json['video_url']
    video_id = video_url.split('v=')[1]
    
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
            
            index.upsert(vectors=upsert_data)
        
        return jsonify({
            "message": "Video processed successfully",
            "video_id": video_id,
            "title": video_info['title'],
            "channel": video_info['uploader'],
            "description": video_info['description'],
            "duration": video_info['duration']
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # Query Pinecone to get relevant context
    query_embedding = retriever.encode([user_message]).tolist()
    query_response = index.query(vector=query_embedding[0], top_k=1, include_metadata=True)
    
    if query_response['matches']:
        context = query_response['matches'][0]['metadata']['chunk_text']
        
        # Generate answer using the fine-tuned model
        answer = generate_answer(user_message, context)
        
        return jsonify({"response": answer})
    else:
        return jsonify({"response": "I'm sorry, I couldn't find relevant information to answer your question."})

if __name__ == '__main__':
    app.run(debug=True)