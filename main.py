import os
import streamlit as st
import pickle
import numpy as np
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import faiss
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.title("News Research Tool 📈 (Simple Version)")
st.sidebar.title("News Article URLs")

def clean_text(text):
    """Clean the extracted text"""
    # Remove citations like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove special characters and symbols
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def is_valid_sentence(sentence):
    """Check if a sentence is valid and meaningful"""
    # Remove empty or very short sentences
    if len(sentence.split()) < 4:
        return False
    # Remove navigation elements and common webpage artifacts
    invalid_starts = ['click', 'search', 'menu', 'sign in', 'subscribe', 
                     'javascript', 'cookie', 'privacy', 'advertisement']
    return not any(sentence.lower().startswith(term) for term in invalid_starts)

def load_url(url):
    """Load and parse content from a URL"""
    try:
        # Add headers to mimic a browser request
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = Request(url, headers=headers)
        html = urlopen(req).read()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                           'sidebar', 'advertisement', 'meta', 'link']):
            element.decompose()
        
        # Focus on main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        # Get all paragraphs
        paragraphs = main_content.find_all('p') if main_content else soup.find_all('p')
        
        # Extract and clean text from paragraphs
        text = ' '.join(p.get_text() for p in paragraphs)
        text = clean_text(text)
        
        return text
    except Exception as e:
        st.error(f"Error loading URL {url}: {str(e)}")
        return ""

def split_into_chunks(text, max_length=1000):
    """Split text into meaningful chunks"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if is_valid_sentence(sentence):
            if current_length + len(sentence) > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:  # Only append non-empty URLs
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "news_research_data.pkl"

if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL before processing.")
    else:
        with st.spinner("Loading and processing articles..."):
            try:
                # Load and process all URLs
                all_chunks = []
                sources = []
                for url in urls:
                    text = load_url(url)
                    if text:
                        chunks = split_into_chunks(text)
                        all_chunks.extend(chunks)
                        sources.extend([url] * len(chunks))
                
                if all_chunks:
                    # Create TF-IDF vectors
                    vectorizer = TfidfVectorizer(stop_words='english')
                    vectors = vectorizer.fit_transform(all_chunks)
                    
                    # Convert to dense vectors for FAISS
                    dense_vectors = vectors.toarray().astype('float32')
                    
                    # Create FAISS index
                    dimension = dense_vectors.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    index.add(dense_vectors)
                    
                    # Save data
                    data = {
                        'index': index,
                        'vectorizer': vectorizer,
                        'texts': all_chunks,
                        'sources': sources
                    }
                    with open(file_path, "wb") as f:
                        pickle.dump(data, f)
                    
                    st.success("✅ Processing complete! You can now ask questions.")
                else:
                    st.error("No content could be extracted from the URLs.")
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

# Query section
if os.path.exists(file_path):
    query = st.text_input("Ask a question about the articles:")
    if query:
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                
            # Transform query
            query_vector = data['vectorizer'].transform([query]).toarray().astype('float32')
            
            # Search similar chunks
            k = 3  # Number of results to return
            D, I = data['index'].search(query_vector, k)
            
            st.header("Relevant Information:")
            
            # Display results
            seen_sources = set()
            for i, idx in enumerate(I[0]):
                if D[0][i] < 20:  # Only show reasonably relevant results
                    text = data['texts'][idx]
                    source = data['sources'][idx]
                    st.markdown(f"**Excerpt {i+1}:**")
                    st.write(text)
                    if source not in seen_sources:
                        seen_sources.add(source)
                        st.caption(f"Source: {source}")
                    st.write("")
                
        except Exception as e:
            st.error(f"Error retrieving information: {str(e)}")
else:
    st.info("👆 Please process some URLs first using the sidebar inputs.")