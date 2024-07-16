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
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()

app = Flask(__name__)

current_video_id = None 

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Global variables
#pc = None
#pinecone_index = None
index_name = 'question-answering'  # Use a fixed name

# Initialize SentenceTransformer
retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

# Define a Hugging Face model to use
HF_MODEL_NAME = "vblagoje/bart_lfqa"

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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

# Call this function at the start of your application
#ensure_pinecone_index()

def load_model():
    print(f"Loading model from Hugging Face: {HF_MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    print("Successfully loaded Hugging Face model")
    return model, tokenizer

# Load the model
model, tokenizer = load_model()

# Create a HuggingFacePipeline
local_llm = HuggingFacePipeline(pipeline=pipeline("text2text-generation", model=model, tokenizer=tokenizer))

# Initialize memory
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
                        'chunk_text': text,
                        'metadata_context': f"Video Title: {info['title']}\nChannel: {info['uploader']}\n"
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
    relevant_contexts = [contexts[i] for i in top_indices if cosine_similarities[i] > 0.05]  
    print(f"Cosine similarities: {cosine_similarities}")
    return relevant_contexts if relevant_contexts else contexts[:top_n]

def generate_answer(question, context):
    max_input_length = 512
    input_text = f"""Answer the question based on the context below. If the question cannot be answered based on the context, say "I don't have enough information to answer that question."

Context: {context[:max_input_length]}

Question: {question}

Answer:"""

    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            max_length=100,  # Shorter to encourage concise answers
            min_length=20, 
            do_sample=False,  # Changed to False for more deterministic outputs
            num_beams=5,  # Using beam search for better quality
            early_stopping=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

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

def get_context(task, video_id, max_contexts=10, is_summarization=False):
    global pinecone_index
    namespace = f"video_{video_id}"
    
    print(f"Retrieving context for video_id: {video_id}, task: {task}, is_summarization: {is_summarization}")
    print(f"Using namespace: {namespace}")
    
    if is_summarization:
        query_response = pinecone_index.query(
            vector=[0]*768,
            top_k=max_contexts,
            include_metadata=True,
            namespace=namespace
        )
    else:
        query_embedding = retriever.encode([task]).tolist()
        query_response = pinecone_index.query(
            vector=query_embedding[0], 
            top_k=max_contexts,
            include_metadata=True, 
            namespace=namespace
        )
    
    print(f"Number of matches: {len(query_response['matches'])}")
    
    if query_response['matches']:
        contexts = [f"{match['metadata']['metadata_context']}\n{match['metadata']['chunk_text']}" for match in query_response['matches']]
        print(f"First context: {contexts[0][:100]}...")  # Print first 100 chars of first context
        return contexts
    return []

def summarize_video(max_summary_length=200):
    global current_video_id
    
    try:
        print(f"Summarizing video with ID: {current_video_id}")
        
        stats = pinecone_index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        contexts = get_context("summarize", current_video_id, max_contexts=10000, is_summarization=True)
        
        if not contexts:
            print("No contexts retrieved for summarization")
            return "I don't have enough information to summarize the video content."
        
        print(f"Number of contexts for summarization: {len(contexts)}")
        all_text = " ".join(contexts)
        print(f"Total text length for summarization: {len(all_text)}")
        
        summary = create_summary(all_text, max_length=max_summary_length)
        print(f"Generated summary: {summary}")
        return f"Here's a summary of the video: {summary}"
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in summarize_video: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return f"An error occurred while trying to summarize the video. Error: {str(e)}"

def create_summary(text, max_length=200):
    if not text:
        return "No text available to summarize."
    
    chunks = chunk_text(text, chunk_size=500)
    summaries = []
    
    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
            print(f"Chunk {i} summarized: {summary[:50]}...")
        except Exception as e:
            print(f"Error summarizing chunk {i}: {str(e)}")
            continue
    
    if not summaries:
        return "Unable to generate a summary."
    
    # Combine summaries and trim to max_length words
    combined_summary = ' '.join(summaries)
    words = combined_summary.split()
    trimmed_summary = ' '.join(words[:max_length])
    
    if len(words) > max_length:
        trimmed_summary += "..."
    
    print(f"Final summary (trimmed to {max_length} words): {trimmed_summary}")
    
    return trimmed_summary.strip()

def determine_task(user_input):
    print(f"Determining task for input: {user_input}")
    
    # List of phrases that indicate a summary request
    summary_phrases = [
        "summarize", "summary", "summarize the video", 
        "what is this video about", "what is the video about", "what is the main topic of the video",
        "what's the video about", "what's the video about", "give me an overview", "brief overview",
        "overview", "main points", "key points", "main idea", "what is the video discussing"
    ]
    
    # Check if the user input contains any of the summary phrases
    if any(phrase in user_input.lower() for phrase in summary_phrases):
        task_type = "summarization"
    else:
        # If not a summary request, assume it's a specific question
        task_type = "question_answering"
    
    print(f"Determined task type: {task_type}")
    return task_type

@app.route('/')
def index():
    return render_template('index.html')

def clear_pinecone_index():
    global pinecone_index
    try:
        stats_before = pinecone_index.describe_index_stats()
        print(f"Index stats before clearing: {stats_before}")
        
        namespaces = stats_before['namespaces'].keys()
        for namespace in namespaces:
            pinecone_index.delete(delete_all=True, namespace=namespace)
        
        stats_after = pinecone_index.describe_index_stats()
        print(f"Index stats after clearing: {stats_after}")
        print("Pinecone index cleared successfully")
    except Exception as e:
        print(f"Error clearing Pinecone index: {str(e)}")

def check_relevance(question, answer, context):
    try:
        relevance_prompt = f"""
        Question: {question}
        Answer: {answer}
        Context: {context[:500]}  # Limit context length

        Is the answer relevant to the question and based on the given context? Respond with Yes or No.
        """
        relevance_check = local_llm(relevance_prompt)
        return relevance_check.strip().lower() == "yes"
    except Exception as e:
        print(f"Error in check_relevance: {str(e)}")
        return True  # Assume relevance if there's an error

@app.route('/process_video', methods=['POST'])
def process_video():
    global pinecone_index, current_video_id

    clear_pinecone_index()
    
    video_url = request.json['video_url']
    try:
        video_id = video_url.split('v=')[1].split('&')[0]  # This will handle URLs with additional parameters
        current_video_id = video_id
        print(f"Processing video with ID: {current_video_id}")
    except IndexError:
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    try:
        # Process the video
        video_data, video_info = preprocess_video(video_id)
        
        # Convert to DataFrame
        df = pd.DataFrame(video_data)
        
        # Upsert to Pinecone
        batch_size = 100
        total_upserted = 0
        for i in tqdm(range(0, len(df), batch_size)):
            i_end = min(i + batch_size, len(df))
            batch = df.iloc[i:i_end]
            emb = retriever.encode(batch['chunk_text'].tolist()).tolist()
            
            upsert_data = [
                (str(row['chunk_id']), vec, row.to_dict())
                for vec, (_, row) in zip(emb, batch.iterrows())
            ]
            
            #print(f"Sample upsert data: {upsert_data[:2]}")
            
            try:
                namespace = f"video_{video_id}"
                upsert_response = pinecone_index.upsert(vectors=upsert_data, namespace=namespace)
                #upsert_response = pinecone_index.upsert(vectors=upsert_data)
                print(f"Upsert response for batch {i//batch_size + 1}: {upsert_response}")
                total_upserted += upsert_response['upserted_count']
            except Exception as e:
                print(f"Error during upsert: {str(e)}")
                print(f"Upsert data sample: {upsert_data[:2]}")
                raise

        print(f"Total vectors upserted: {total_upserted}")

        # Check if data was inserted successfully
        time.sleep(5)
        stats = pinecone_index.describe_index_stats()
        namespace_stats = stats['namespaces'].get(namespace, {})
        print(f"Namespace '{namespace}' stats: {namespace_stats}")

        if stats['total_vector_count'] != total_upserted:
            print(f"Warning: Mismatch between upserted count ({total_upserted}) and total vector count ({stats['total_vector_count']})")

        return jsonify({
            "message": "Video processed successfully",
            "video_id": video_id,
            "title": video_info['title'],
            "channel": video_info['uploader'],
            "description": video_info['description'],
            "duration": video_info['duration'],
            "index_name": index_name,
            "vectors_inserted": total_upserted
        })
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error processing video: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global pinecone_index, current_video_id
    
    user_message = request.json['message'].strip()
    print(f"Received user message: {user_message}")
    print(f"Current video ID: {current_video_id}")
    
    if current_video_id is None:
        return jsonify({"error": "No video has been processed yet. Please process a video first."}), 400

    try:
        task_type = determine_task(user_message)
        print(f"Determined task type: {task_type}")
        
        if task_type == 'summarization':
            summary = summarize_video(max_summary_length=200)  
            return jsonify({"response": summary, "context": ""})
        
        else:  # question_answering
            contexts = get_context(user_message, current_video_id, max_contexts=3, is_summarization=False)
            print(f"Retrieved contexts: {contexts}")
            
            if not contexts:
                response = "I don't have enough information to answer that question based on the video content."
            else:
                combined_context = " ".join(contexts)
                answer = generate_answer(user_message, combined_context)
                
                if "I don't have enough information" in answer:
                    response = f"I couldn't find a specific answer to your question. However, here's some relevant information from the video: {combined_context[:200]}..."
                else:
                    response = answer
            
            print(f"Generated response: {response}")
            return jsonify({
                "response": response,
                "context": combined_context[:500] + "..." if len(combined_context) > 500 else combined_context
            })
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in chat: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400

if __name__ == '__main__':
    app.run(debug=True)