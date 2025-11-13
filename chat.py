import streamlit as st
import PyPDF2
import re
import os
from sentence_transformers import SentenceTransformer, util
import torch

# Set page configuration
st.set_page_config(
    page_title="Quranic Guidance Chatbot",
    page_icon="📖",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .verse-box {
        background-color: #f8f9fa;
        border-left: 5px solid #1f4e79;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .surah-info {
        color: #1f4e79;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .response-text {
        font-size: 1.1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📖 Quranic Guidance Chatbot</h1>', unsafe_allow_html=True)
st.write("Share your feelings, problems, or questions, and receive relevant guidance from the Holy Quran.")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_quran_data():
    """Load and parse the Quran PDF content"""
    try:
        # Read the PDF file
        pdf_path = "Holy-Quran-English.pdf"
        
        if not os.path.exists(pdf_path):
            st.error(f"Quran PDF file not found at: {pdf_path}")
            return []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            verses = []
            current_chapter = None
            current_verse_num = 0
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Split text into lines and process
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    
                    # Skip empty lines and headers
                    if not line or 'THE HOLY QUR' in line.upper() or 'CHAPTER' in line.upper():
                        continue
                    
                    # Look for chapter headers (like "AL-FATIHAH", "AL-BAQARAH")
                    if (line.isupper() and len(line) > 3 and 
                        not line.startswith('Part') and 
                        not line.startswith('R.')):
                        current_chapter = line
                        current_verse_num = 0
                        continue
                    
                    # Look for verse patterns - numbers followed by text
                    verse_match = re.match(r'^(\d+)\.\s+(.+)$', line)
                    if verse_match and current_chapter:
                        current_verse_num = verse_match.group(1)
                        verse_text = verse_match.group(2)
                        
                        # Clean up the verse text
                        verse_text = re.sub(r'[†*‡]', '', verse_text)  # Remove footnotes markers
                        verse_text = re.sub(r'\s+', ' ', verse_text).strip()
                        
                        if len(verse_text) > 10:  # Ensure it's a meaningful verse
                            verses.append({
                                'chapter': current_chapter,
                                'verse_number': current_verse_num,
                                'text': verse_text,
                                'full_reference': f"{current_chapter} {current_verse_num}"
                            })
            
            return verses
            
    except Exception as e:
        st.error(f"Error loading Quran data: {str(e)}")
        return []

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model for semantic similarity"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

def find_relevant_verses(query, verses, model, top_k=5):
    """Find the most relevant Quranic verses for the query"""
    if not verses or not model:
        return []
    
    try:
        # Encode the query and all verses
        query_embedding = model.encode([query], convert_to_tensor=True)
        verse_texts = [verse['text'] for verse in verses]
        verse_embeddings = model.encode(verse_texts, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = util.pytorch_cos_sim(query_embedding, verse_embeddings)[0]
        
        # Get top k most similar verses
        top_indices = torch.topk(similarities, min(top_k, len(verses))).indices
        
        relevant_verses = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Similarity threshold
                verse_data = verses[idx.item()].copy()
                verse_data['similarity_score'] = similarities[idx].item()
                relevant_verses.append(verse_data)
        
        return relevant_verses
        
    except Exception as e:
        st.error(f"Error finding relevant verses: {str(e)}")
        return []

# Emotion to Quranic theme mapping
EMOTION_KEYWORDS = {
    'sad': ['patience', 'comfort', 'hope', 'mercy', 'difficulty', 'relief'],
    'anxious': ['patience', 'trust', 'peace', 'fear', 'worry', 'calm'],
    'angry': ['forgiveness', 'patience', 'control', 'anger', 'peace'],
    'happy': ['gratitude', 'thanks', 'blessings', 'joy', 'happiness'],
    'confused': ['guidance', 'wisdom', 'knowledge', 'understanding', 'clarity'],
    'scared': ['protection', 'safety', 'trust', 'fear', 'courage'],
    'lonely': ['companionship', 'comfort', 'mercy', 'love', 'solitude'],
    'stressed': ['peace', 'patience', 'relief', 'ease', 'burden'],
    'grateful': ['thanks', 'blessings', 'gratitude', 'favors', 'appreciation'],
    'hopeless': ['hope', 'mercy', 'relief', 'help', 'despair']
}

def enhance_query_with_emotion(query):
    """Enhance the query with emotion-related keywords"""
    query_lower = query.lower()
    enhanced_query = query
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if emotion in query_lower:
            enhanced_query += " " + " ".join(keywords)
            break
    
    return enhanced_query

def display_verse(verse_data):
    """Display a verse in a formatted box"""
    st.markdown(f"""
    <div class="verse-box">
        <div class="surah-info">{verse_data['full_reference']}</div>
        <div class="response-text">{verse_data['text']}</div>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    # Load data and models
    with st.spinner("Loading Quranic data..."):
        verses = load_quran_data()
        model = load_embedding_model()
    
    if not verses:
        st.error("Could not load Quranic verses. Please check the PDF file.")
        return
    
    st.success(f"✅ Loaded {len(verses)} Quranic verses")
    
    # Example queries
    st.markdown("### 💡 Example Queries:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("I'm feeling sad"):
            st.session_state.user_query = "I'm feeling sad and need comfort"
    with col2:
        if st.button("I'm anxious"):
            st.session_state.user_query = "I'm feeling anxious and worried"
    with col3:
        if st.button("I need guidance"):
            st.session_state.user_query = "I need guidance and direction"
    
    # User input
    user_query = st.text_area(
        "Share your feelings or questions:",
        value=st.session_state.get('user_query', ''),
        placeholder="e.g., 'I'm feeling sad today', 'I need guidance about my future', 'How to deal with anger?'",
        height=100
    )
    
    if st.button("Get Quranic Guidance", type="primary"):
        if user_query:
            with st.spinner("Finding relevant Quranic verses..."):
                # Enhance query with emotion keywords
                enhanced_query = enhance_query_with_emotion(user_query)
                
                # Find relevant verses
                relevant_verses = find_relevant_verses(enhanced_query, verses, model, top_k=5)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'query': user_query,
                    'verses': relevant_verses,
                    'timestamp': len(st.session_state.chat_history)
                })
                
                # Display results
                if relevant_verses:
                    st.markdown("### 📜 Relevant Quranic Verses:")
                    
                    for verse in relevant_verses:
                        display_verse(verse)
                    
                    st.markdown("---")
                    st.markdown("""
                    <div style='text-align: center; color: #666; font-style: italic;'>
                        May these verses bring you peace and guidance. Reflect upon them with an open heart.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("""
                    No specific verses found for your query. Here are some general verses of comfort:
                    """)
                    # Fallback to general comforting verses
                    comfort_verses = [
                        v for v in verses 
                        if any(keyword in v['text'].lower() for keyword in ['mercy', 'comfort', 'patience', 'peace'])
                    ][:3]
                    
                    for verse in comfort_verses:
                        display_verse(verse)
        
        else:
            st.warning("Please share your feelings or questions to receive Quranic guidance.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 Your Recent Queries")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:]), 1):
            with st.expander(f"Query {len(st.session_state.chat_history)-i+1}: {chat['query'][:50]}..."):
                st.write(f"**Your question:** {chat['query']}")
                st.write("**Guidance:**")
                for verse in chat['verses'][:3]:
                    display_verse(verse)

if __name__ == "__main__":
    main()
