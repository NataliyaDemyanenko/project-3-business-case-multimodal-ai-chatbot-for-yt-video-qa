# YouTube Video Q&A and summarization ChatBot

## Project Overview

This project implements a multimodal AI ChatBot capable of summarizing YouTube videos and answering questions about their content. The chatbot uses advanced natural language processing techniques to provide accurate and coherent responses to user queries.

### Key Features

- YouTube video content extraction and processing
- Abstractive summarization of video content
- Question-answering based on video content
- Hybrid context retrieval for improved accuracy
- Post-processing for enhanced readability

## Repository Structure

- `Report.pdf`: Detailed project report
- `Final project presentation.pdf`: Project presentation slides
- `Deployment/`: Folder containing the final app suitable for external hosting
  - `app.py`: Main application file
  - `templates/`: Folder containing HTML templates
    - `index.html`: Main page template
  - `requirements.txt`: List of required Python packages
  - `Dockerfile`: Configuration for Docker deployment
- `Basic Deployment/`: Initial deployment version
- `Deployment with bart for summarization and quantized model/`: Intermediate deployment version
- `Fine-tuning and evaluation/`: Contains files with fine-tuned models and evaluation results
- `Preprocessing/`: Contains files detailing preprocessing steps

## Technology Stack

- Python
- Flask (Web framework)
- Docker
- TensorFlow and PyTorch
- Hugging Face Transformers
- Pinecone (Vector database)
- Langchain
- NLTK (Natural Language Toolkit)
- yt-dlp and youtube_transcript_api (for YouTube video processing)

## Installation and Setup

1. Clone the repository:
git clone [repository-url]
cd [repository-name]

2. Install required packages:
pip install -r Deployment/requirements.txt

3. Set up environment variables:
OPENAI_API_KEY, PINECONE_API_KEY

5. Run the application:
python Deployment/app.py

## Docker Deployment

To deploy using Docker:

1. Build the Docker image:
docker build -t youtube-qa-bot Deployment/
2. Run the Docker container:
docker run -p 5000:5000 youtube-qa-bot

## Usage

1. Access the application through a web browser at `http://localhost:5000`
2. Enter a YouTube video URL
3. Choose between summarization or ask a specific question about the video content
4. Receive the generated summary or answer

## Key Components

### Initialization
- `load_model()`: Loads the pre-trained BART-LFQA model
- `clear_pinecone_index()`: Clears the Pinecone index for new video processing

### Video Processing
- `preprocess_video()`: Extracts video information and transcript
- `extract_key_topics()`: Identifies main topics from video title and description

### Task Determination
- `determine_task()`: Decides between summarization and question-answering based on user input

### Summarization
- `get_context()`: Retrieves relevant context for summarization
- `abstractive_summarize()`: Generates a concise summary using the BART model
- `post_process_summary()`: Refines the generated summary for readability

### Question-Answering
- `generate_answer()`: Produces answers based on retrieved context
- `check_relevance()`: Verifies answer relevance
- `hybrid_context_retrieval()`: Combines TF-IDF and semantic similarity for context retrieval

### Common Functions
- `remove_repetitions()`: Eliminates duplicate sentences
- `split_into_sentences()`: Properly formats text into sentences
- `clean_and_format_sentence()`: Ensures consistent sentence formatting

## Model Details

The project uses the VBlagoje/BART-LFQA model, chosen for its advanced question-answering capabilities, state-of-the-art architecture, and optimization for long-form answers.

## Future Work

- Process videos without relying on YouTube transcripts
- Fine-tune both question-answering and summarization models
- Implement more advanced sentence reordering in summarization
- Integrate more sophisticated NLP techniques for context extraction and answer generation

## Contributors

Nataliya Demyanenko

## License

This project is licensed under the MIT License.

Copyright (c) 2024 Nataliya Demyanenko

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.