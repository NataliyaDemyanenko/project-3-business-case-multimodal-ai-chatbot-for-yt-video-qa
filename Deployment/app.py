import os
import time
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from langchain_community.llms import HuggingFacePipeline
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoConfig, BartForConditionalGeneration, BartTokenizer
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from collections import Counter
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import OrderedDict

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load environment variables
load_dotenv()

app = Flask(__name__)

current_video_id = None 

# Initialize SentenceTransformer
retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

# Define a Hugging Face model to use
HF_MODEL_NAME = "vblagoje/bart_lfqa"

# Load the model and tokenizer
summarization_model_name = "facebook/bart-large-cnn"
summarization_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)
summarization_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)

# Define the index name
index_name = 'question-answering'

# Global variables for lazy loading
model_instance = None
tokenizer_instance = None
pinecone_client = None
pinecone_index = None

@lru_cache(maxsize=1)
def get_pinecone_client():
    global pinecone_client
    if pinecone_client is None:
        pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    return pinecone_client

@lru_cache(maxsize=1)
def get_pinecone_index():
    global pinecone_index
    if pinecone_index is None:
        client = get_pinecone_client()
        if index_name not in client.list_indexes().names():
            client.create_index(
                name=index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
        pinecone_index = client.Index(index_name)
    return pinecone_index

def unload_pinecone():
    global pinecone_client, pinecone_index
    pinecone_client = None
    pinecone_index = None
    get_pinecone_client.cache_clear()
    get_pinecone_index.cache_clear()

def get_model():
    global model_instance, tokenizer_instance
    if model_instance is None or tokenizer_instance is None:
        print(f"Loading model from Hugging Face: {HF_MODEL_NAME}")
        model_instance = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
        tokenizer_instance = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        print("Successfully loaded Hugging Face model")
    return model_instance, tokenizer_instance

def unload_model():
    global model_instance, tokenizer_instance
    model_instance = None
    tokenizer_instance = None

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
                        'metadata_context': f"Video Title: {info['title']}\n"
                    })
                combined_text = ""
                combined_start_time = 0
                combined_duration = 0
        print("Transcript processing completed")

        return video_data, info
    except Exception as e:
        print(f"Error in preprocess_video: {str(e)}")
        raise

def get_most_relevant_context(question, contexts, top_n=2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([question] + contexts)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    question_embedding = retriever.encode([question])
    context_embeddings = retriever.encode(contexts)
    semantic_scores = util.pytorch_cos_sim(question_embedding, context_embeddings).numpy().flatten()
    
    tfidf_scores = np.array(tfidf_scores)
    semantic_scores = np.array(semantic_scores)
    
    combined_scores = 0.5 * tfidf_scores + 0.5 * semantic_scores
    
    top_indices = combined_scores.argsort()[-top_n:][::-1]
    return [contexts[i] for i in top_indices]

def remove_duplicates(text):
    sentences = sent_tokenize(text)
    unique_sentences = list(OrderedDict.fromkeys(sentences))
    return ' '.join(unique_sentences)

def clean_and_format_sentence(sentence):
    sentence = re.sub(r'\s+', ' ', sentence).strip().rstrip(',')
    sentence = re.sub(r'\[Music\]', '', sentence)
    
    if not sentence.endswith(('.', '?', '!')):
        sentence += '.'

    if len(sentence) > 0:
        sentence = sentence[0].upper() + sentence[1:]

    return sentence

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    cleaned_sentences = [clean_and_format_sentence(s) for s in sentences if len(s.strip()) > 10]
    return ' '.join(cleaned_sentences)

def generate_answer(question, context):
    max_input_length = 1024
    input_text = f"""Answer the question based on the following context. If the question cannot be answered based on the context, respond only with "I don't have information about that in the context of this video."

Context: {context[:max_input_length]}

Question: {question}

Answer:"""

    model, tokenizer = get_model()
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            max_length=300,
            min_length=50,
            do_sample=True,
            num_beams=5,
            top_p=0.95,
            temperature=0.7,
            early_stopping=True,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "I don't have information about that in the context of this video" in answer:
        return ["I don't have information about that in the context of this video."]
    else:
        return split_into_sentences(answer)

def extract_relevant_sentences(question, context, num_sentences=2):
    sentences = sent_tokenize(context)
    
    stop_words = set(stopwords.words('english'))
    question_words = [w.lower() for w in word_tokenize(question) if w.lower() not in stop_words]
    
    scores = []
    for sentence in sentences:
        words = [w.lower() for w in word_tokenize(sentence) if w.lower() not in stop_words]
        score = sum(1 for w in question_words if w in words)
        scores.append(score)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
    top_sentences = [sentences[i] for i in sorted(top_indices)]
    
    return " ".join(top_sentences)

def remove_repetitions(text):
    sentences = text.split('.')
    unique_sentences = []
    for sentence in sentences:
        if sentence.strip() and sentence.strip() not in unique_sentences:
            unique_sentences.append(sentence.strip())
    return '. '.join(unique_sentences) + '.'

def hybrid_context_retrieval(query, contexts, top_k=7, tfidf_weight=0.4):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([query] + contexts)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    query_embedding = retriever.encode([query])
    context_embeddings = retriever.encode(contexts)
    semantic_scores = cosine_similarity(query_embedding, context_embeddings)[0]
    
    combined_scores = tfidf_weight * tfidf_scores + (1 - tfidf_weight) * semantic_scores
    
    top_indices = combined_scores.argsort()[-top_k:][::-1]
    return [contexts[i] for i in top_indices]

def get_context(task, video_id, max_contexts=20, is_summarization=False):
    pinecone_index = get_pinecone_index()
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
            top_k=max_contexts * 2,
            include_metadata=True, 
            namespace=namespace
        )
    
    print(f"Number of matches: {len(query_response['matches'])}")
    
    if query_response['matches']:
        contexts = [f"{match['metadata']['metadata_context']}\n{match['metadata']['chunk_text']}" for match in query_response['matches']]
        
        if not is_summarization:
            contexts = hybrid_context_retrieval(task, contexts, top_k=max_contexts)
        
        print(f"First context: {contexts[0][:100]}...")
        return contexts
    return []

def extract_key_topics(title, description):
    text = f"{title} {description}".lower()
    words = re.findall(r'\w+', text)
    word_counts = Counter(words)
    
    stop_words = set(stopwords.words('english'))
    key_topics = [word for word, count in word_counts.most_common(5) if word not in stop_words]
    
    return key_topics

def abstractive_summarize(text, key_topics, max_length=150, min_length=50):
    topic_text = " ".join(key_topics)
    full_text = f"{topic_text}. {text}"
    
    inputs = summarization_tokenizer([full_text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = summarization_model.generate(
        inputs["input_ids"], 
        num_beams=4, 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=1.5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def add_transitions(sentences):
    transitions = [
        "Additionally, ", "Furthermore, ", "Moreover, ", "In addition, ",
        "Also, ", "Besides this, ", "What's more, "
    ]
    for i in range(1, len(sentences)):
        if random.random() < 0.3:
            sentences[i] = random.choice(transitions) + sentences[i][0].lower() + sentences[i][1:]
    return sentences

def fix_capitalization(text):
    sentences = sent_tokenize(text)
    capitalized_sentences = [s.capitalize() for s in sentences]
    
    text = ' '.join(capitalized_sentences)
    
    words = word_tokenize(text)
    tagged = pos_tag(words)
    
    capitalized_words = []
    for word, tag in tagged:
        if tag.startswith('NNP'):
            capitalized_words.append(word.capitalize())
        else:
            capitalized_words.append(word)
    
    return ' '.join(capitalized_words)

def post_process_summary(summary, key_topics):
    summary = re.sub(r'Video title:.*?\.\s*', '', summary, flags=re.IGNORECASE)
    
    sentences = sent_tokenize(summary)
    
    sentences = [s for s in sentences if len(s.split()) > 5]
    
    sentences = add_transitions(sentences)
    
    processed_summary = ' '.join(sentences)
    
    processed_summary = fix_capitalization(processed_summary)

    processed_summary = re.sub(r'\s+\.', '.', processed_summary)

    processed_summary = re.sub(r'\s+,', ',', processed_summary)
    
    return processed_summary

def summarize_video(max_summary_length=250):
    global current_video_id
    
    try:
        print(f"Summarizing video with ID: {current_video_id}")
        
        contexts = get_context("summarize", current_video_id, max_contexts=10000, is_summarization=True)
        
        if not contexts:
            print("No contexts retrieved for summarization")
            return "Insufficient information to summarize the video content."
        
        video_info = contexts[0].split('\n', 1)
        video_title = video_info[0].replace("Video Title: ", "")
        video_description = video_info[1] if len(video_info) > 1 else ""
        
        key_topics = extract_key_topics(video_title, video_description)
        
        print(f"Number of contexts for summarization: {len(contexts)}")
        all_text = " ".join(contexts)
        print(f"Total text length for summarization: {len(all_text)}")
        
        final_summary = abstractive_summarize(all_text, key_topics, max_length=max_summary_length, min_length=100)
        
        final_summary = post_process_summary(final_summary, key_topics)
        
        print(f"Generated summary: {final_summary}")
        return f"Video summary: {final_summary}"
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in summarize_video: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return f"An error occurred while summarizing the video: {str(e)}"

def determine_task(user_input):
    print(f"Determining task for input: {user_input}")
    
    summary_phrases = [
        "summarize", "summary", "summarize the video", 
        "what is this video about", "what is the video about", "what is the main topic of the video",
        "what's the video about", "what's the video about", "give me an overview", "brief overview",
        "overview", "main points", "key points", "main idea", "what is the video discussing"
    ]
    
    if any(phrase in user_input.lower() for phrase in summary_phrases):
        task_type = "summarization"
    else:
        task_type = "question_answering"
    
    print(f"Determined task type: {task_type}")
    return task_type

@app.route('/')
def index():
    return render_template('index.html')

def clear_pinecone_index():
    pinecone_index = get_pinecone_index()
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

@app.route('/process_video', methods=['POST'])
def process_video():
    global current_video_id
    pinecone_index = get_pinecone_index()

    clear_pinecone_index()
    
    video_url = request.json['video_url']
    try:
        video_id = video_url.split('v=')[1].split('&')[0]
        current_video_id = video_id
        print(f"Processing video with ID: {current_video_id}")
    except IndexError:
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    try:
        video_data, video_info = preprocess_video(video_id)
        df = pd.DataFrame(video_data)
        
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
            
            try:
                namespace = f"video_{video_id}"
                upsert_response = pinecone_index.upsert(vectors=upsert_data, namespace=namespace)
                print(f"Upsert response for batch {i//batch_size + 1}: {upsert_response}")
                total_upserted += upsert_response['upserted_count']
            except Exception as e:
                print(f"Error during upsert: {str(e)}")
                print(f"Upsert data sample: {upsert_data[:2]}")
                raise

        print(f"Total vectors upserted: {total_upserted}")

        time.sleep(5)
        stats = pinecone_index.describe_index_stats()
        namespace_stats = stats['namespaces'].get(f"video_{video_id}", {})
        print(f"Namespace 'video_{video_id}' stats: {namespace_stats}")

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
    global current_video_id
    
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
            contexts = get_context(user_message, current_video_id, max_contexts=10, is_summarization=False)
            
            if not contexts:
                response = ["I don't have specific information to answer that question based on the video content."]
            else:
                combined_context = " ".join(contexts)
                print(f"Combined context: {combined_context[:500]}...")
                answer = generate_answer(user_message, combined_context)
                response = answer
            
            print(f"Generated response: {response}")
            return jsonify({
                "response": response,
                "context": combined_context[:1000] + "..." if len(combined_context) > 1000 else combined_context
            })
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in chat: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 400
    finally:
        unload_model()
        unload_pinecone()
        
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv("PORT", default=5000))