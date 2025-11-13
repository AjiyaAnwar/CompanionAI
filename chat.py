import streamlit as st
import PyPDF2
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page configuration
st.set_page_config(
    page_title="Quranic Guidance Chatbot",
    page_icon="📖",
    layout="centered"
)

# Custom CSS for better styling with BLACK TEXT for verses
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
        font-size: 1.1rem;
    }
    .response-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #000000 !important;  /* Black text for verses */
    }
    .stButton button {
        width: 100%;
    }
    /* Ensure all text in verse boxes is black */
    .verse-box * {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📖 Quranic Guidance Chatbot</h1>', unsafe_allow_html=True)
st.write("Share your feelings, problems, or questions, and receive relevant guidance from the Holy Quran.")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'verses' not in st.session_state:
    st.session_state.verses = []

# Sample Quranic verses database (fallback if PDF parsing fails)
SAMPLE_VERSES = [
    {
        'chapter': 'Al-Baqarah',
        'verse_number': '286',
        'text': 'Allah does not burden any soul beyond its capacity. It shall have the reward it earns, and it shall get the punishment it incurs.',
        'full_reference': 'Al-Baqarah 286',
        'keywords': ['sad', 'stressed', 'burden', 'difficulty', 'capacity', 'test']
    },
    {
        'chapter': 'Ash-Sharh', 
        'verse_number': '5-6',
        'text': 'So, surely, with every difficulty there is relief. Surely, with every difficulty there is relief.',
        'full_reference': 'Ash-Sharh 5-6',
        'keywords': ['difficulty', 'relief', 'stress', 'problem', 'solution', 'hope']
    },
    {
        'chapter': 'Az-Zumar',
        'verse_number': '53',
        'text': 'Say, "O My servants who have transgressed against themselves, do not despair of the mercy of Allah. Indeed, Allah forgives all sins."',
        'full_reference': 'Az-Zumar 53',
        'keywords': ['despair', 'mercy', 'forgiveness', 'hope', 'sad', 'regret']
    },
    {
        'chapter': 'Ar-Raad',
        'verse_number': '28',
        'text': 'Those who believe and whose hearts find comfort in the remembrance of Allah. Aye! it is in the remembrance of Allah that hearts can find comfort.',
        'full_reference': 'Ar-Raad 28',
        'keywords': ['comfort', 'peace', 'anxious', 'worried', 'remembrance', 'heart']
    },
    {
        'chapter': 'Al-Baqarah',
        'verse_number': '186',
        'text': 'And when My servants ask thee about Me, say: I am near. I answer the prayer of the supplicant when he prays to Me.',
        'full_reference': 'Al-Baqarah 186',
        'keywords': ['prayer', 'help', 'near', 'answer', 'supplication', 'request']
    },
    {
        'chapter': 'Ali-Imran',
        'verse_number': '159',
        'text': 'So by mercy from Allah, you were gentle with them. And if you had been rude in speech and harsh in heart, they would have disbanded from about you.',
        'full_reference': 'Ali-Imran 159',
        'keywords': ['gentle', 'mercy', 'anger', 'patience', 'kindness', 'forgiveness']
    },
    {
        'chapter': 'Al-Hijr',
        'verse_number': '49',
        'text': 'Tell My servants that I am the Forgiving, the Merciful.',
        'full_reference': 'Al-Hijr 49',
        'keywords': ['forgiving', 'merciful', 'hope', 'repentance', 'mercy']
    },
    {
        'chapter': 'Ibrahim',
        'verse_number': '7',
        'text': 'And when your Lord proclaimed: If you are grateful, I will surely give you more; but if you are ungrateful, My punishment is indeed severe.',
        'full_reference': 'Ibrahim 7',
        'keywords': ['grateful', 'thanks', 'blessings', 'happy', 'appreciation']
    },
    {
        'chapter': 'An-Nahl',
        'verse_number': '97',
        'text': 'Whoever works righteousness, whether male or female, and is a believer, We will surely grant him a pure life; and We will surely give them their reward according to the best of their works.',
        'full_reference': 'An-Nahl 97',
        'keywords': ['righteousness', 'reward', 'pure life', 'believer', 'good deeds']
    },
    {
        'chapter': 'Ta-Ha',
        'verse_number': '2-3',
        'text': 'We have not sent down the Quran to thee that thou shouldst be distressed, But as an admonition to him who fears.',
        'full_reference': 'Ta-Ha 2-3',
        'keywords': ['distress', 'admonition', 'fear', 'guidance', 'comfort']
    }
]

def load_quran_data():
    """Load Quran data - try PDF first, then use samples"""
    try:
        # Try to find PDF file
        pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
        
        if pdf_files:
            pdf_path = pdf_files[0]
            st.info(f"Found PDF: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                st.success(f"PDF loaded with {len(pdf_reader.pages)} pages")
                
                # For now, return sample verses
                # You can implement PDF parsing here later
                return SAMPLE_VERSES
        else:
            st.info("No PDF found. Using sample Quranic verses.")
            return SAMPLE_VERSES
            
    except Exception as e:
        st.warning(f"Using sample verses due to: {str(e)}")
        return SAMPLE_VERSES

def find_relevant_verses_simple(query, verses, top_k=5):
    """Simple keyword-based search for relevant verses"""
    query_lower = query.lower()
    scored_verses = []
    
    for verse in verses:
        score = 0
        
        # Score based on keyword matches
        if 'keywords' in verse:
            for keyword in verse['keywords']:
                if keyword in query_lower:
                    score += 2
        
        # Score based on direct word matches in verse text
        verse_text_lower = verse['text'].lower()
        query_words = [word for word in query_lower.split() if len(word) > 3]
        
        for word in query_words:
            if word in verse_text_lower:
                score += 1
        
        # Bonus for emotional words
        emotional_words = ['sad', 'happy', 'angry', 'anxious', 'worried', 'stressed', 'lonely', 'scared', 'confused']
        for emotion in emotional_words:
            if emotion in query_lower and emotion in verse_text_lower:
                score += 3
        
        if score > 0:
            scored_verses.append((verse, score))
    
    # Sort by score and return top results
    scored_verses.sort(key=lambda x: x[1], reverse=True)
    
    if scored_verses:
        return [verse for verse, score in scored_verses[:top_k]]
    else:
        # Return some comforting verses if no matches found
        return [v for v in verses if any(kw in ['comfort', 'mercy', 'patience'] for kw in v.get('keywords', []))][:top_k]

def display_verse(verse_data):
    """Display a verse in a formatted box with BLACK TEXT"""
    st.markdown(f"""
    <div class="verse-box">
        <div class="surah-info">{verse_data['full_reference']}</div>
        <div class="response-text">{verse_data['text']}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Load Quran data
    if not st.session_state.verses:
        with st.spinner("Loading Quranic verses..."):
            st.session_state.verses = load_quran_data()
    
    if st.session_state.verses:
        st.success(f"✅ Ready with {len(st.session_state.verses)} Quranic verses")
    
    # Emotion buttons
    st.markdown("### 💡 How are you feeling today?")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("😔 Sad", use_container_width=True):
            st.session_state.user_query = "I'm feeling sad and need comfort and hope"
    with col2:
        if st.button("😰 Anxious", use_container_width=True):
            st.session_state.user_query = "I'm feeling anxious and worried about the future"
    with col3:
        if st.button("🧭 Lost", use_container_width=True):
            st.session_state.user_query = "I feel lost and need guidance in life"
    
    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("😠 Angry", use_container_width=True):
            st.session_state.user_query = "I'm feeling angry and need patience and control"
    with col5:
        if st.button("🏻 Lonely", use_container_width=True):
            st.session_state.user_query = "I'm feeling lonely and isolated from others"
    with col6:
        if st.button("😥 Stressed", use_container_width=True):
            st.session_state.user_query = "I'm feeling stressed and overwhelmed with life"
    
    # More emotion buttons
    col7, col8, col9 = st.columns(3)
    with col7:
        if st.button("😊 Grateful", use_container_width=True):
            st.session_state.user_query = "I'm feeling grateful and thankful for blessings"
    with col8:
        if st.button("😟 Scared", use_container_width=True):
            st.session_state.user_query = "I'm feeling scared and need protection and courage"
    with col9:
        if st.button("🤔 Confused", use_container_width=True):
            st.session_state.user_query = "I'm feeling confused and need clarity and wisdom"
    
    # User input
    user_query = st.text_area(
        "Or describe your feelings in your own words:",
        value=st.session_state.get('user_query', ''),
        placeholder="e.g., 'I'm feeling sad today', 'I need guidance about my future', 'How to deal with anger?', 'I feel anxious about my relationships'",
        height=100
    )
    
    if st.button("📖 Get Quranic Guidance", type="primary", use_container_width=True):
        if user_query:
            with st.spinner("Finding relevant Quranic verses for you..."):
                # Find relevant verses using simple keyword matching
                relevant_verses = find_relevant_verses_simple(user_query, st.session_state.verses, top_k=5)
                
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
                    st.info("Here are some comforting Quranic verses:")
                    for verse in st.session_state.verses[:3]:
                        display_verse(verse)
        
        else:
            st.warning("Please share your feelings or questions to receive Quranic guidance.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 Your Recent Queries")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:]), 1):
            with st.expander(f"Query {len(st.session_state.chat_history)-i+1}: {chat['query'][:50]}..."):
                st.write(f"**Your feeling:** {chat['query']}")
                st.write("**Quranic guidance:**")
                for verse in chat['verses'][:3]:
                    display_verse(verse)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "This chatbot provides guidance from the Holy Quran. For specific religious rulings, please consult qualified scholars."
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
