import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import re
import os

# Initialize Streamlit app
st.title("Quranic Guidance Chatbot")
st.write("Ask about your emotions, problems, or confusions, and find relevant Quranic verses.")

# Step 1: Load and preprocess the Quran PDF
@st.cache_resource
def load_knowledge_base():
    file_path = "The-Quran-Saheeh-International.pdf"
    if not os.path.exists(file_path):
        st.error("Please ensure 'The-Quran-Saheeh-International.pdf' is in the project directory.")
        return None, None
    
    # Extract text from PDF
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text() or ""
        text += extracted_text + "\n"
    
    # Split text into verses, assuming format like "1:1 Text" or "Surah 1, Ayah 1: Text"
    documents = []
    metadata = []
    lines = text.split("\n")
    for line in lines:
        match = re.match(r"(\d+):(\d+)\s+(.+)", line.strip()) or \
                re.match(r"Surah\s+(\d+),\s+Ayah\s+(\d+):\s+(.+)", line.strip(), re.IGNORECASE)
        if match:
            surah, ayah, verse_text = match.groups()
            documents.append(verse_text.strip())
            metadata.append({"surah": surah, "ayah": ayah})
    
    if not documents:
        st.error("No verses found. Ensure the PDF has Surah:Ayah numbers with text.")
        return None, None
    
    # Split long verses if needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    chunk_metadata = []
    for i, doc in enumerate(documents):
        split_chunks = text_splitter.split_text(doc)
        chunks.extend(split_chunks)
        chunk_metadata.extend([metadata[i]] * len(split_chunks))
    
    return chunks, chunk_metadata

# Step 2: Create embeddings and vector store
@st.cache_resource
def create_vector_store(chunks, metadata):
    if chunks is None:
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model,
        metadatas=metadata
    )
    return vector_store

# Step 3: Initialize the LLM
@st.cache_resource
def load_llm():
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",  # Lightweight for CPU
        task="text-generation",
        pipeline_kwargs={"max_length": 200, "truncation": True}
    )
    return llm

# Step 4: Set up prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="You are a respectful assistant providing guidance based on the Quran (Sahih International translation). The user has expressed: '{query}'. Based on the following Quranic verses (with Surah and Ayah numbers):\n{context}\n\nProvide a concise, comforting, and accurate response, citing the relevant verses (e.g., Surah Al-Baqarah, 2:286). Avoid speculation and focus on the provided context."
)

# Step 5: Main chatbot logic
def get_response(query, vector_store, llm):
    if vector_store is None or not query:
        return "Please provide a valid query and ensure the Quran PDF is loaded.", ""
    
    # Retrieve relevant verses
    relevant_docs = vector_store.similarity_search_with_score(query, k=3)
    context = ""
    for doc, score in relevant_docs:
        surah = doc.metadata["surah"]
        ayah = doc.metadata["ayah"]
        context += f"Surah {surah}, Ayah {ayah}: {doc.page_content}\n"
    
    # Generate response
    prompt = prompt_template.format(context=context, query=query)
    response = llm(prompt)
    return response, context

# Load resources
chunks, metadata = load_knowledge_base()
vector_store = create_vector_store(chunks, metadata)
llm = load_llm()

# Step 6: Streamlit interface
query = st.text_input("Share your emotion, problem, or confusion (e.g., 'I feel sad' or 'What does the Quran say about patience?'):")
if query:
    with st.spinner("Retrieving Quranic verses..."):
        response, context = get_response(query, vector_store, llm)
        st.write("**Response:**")
        st.write(response)
        with st.expander("Relevant Quranic Verses"):
            st.write(context)
