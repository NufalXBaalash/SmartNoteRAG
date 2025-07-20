import streamlit as st
import tempfile
import os
import sys

# Check and install dependencies
def check_dependencies():
    required_packages = {
        'sentence_transformers': 'sentence-transformers',
        'google.generativeai': 'google-generativeai',
        'pypdf': 'pypdf',
        'torch': 'torch'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        st.error(f"Missing required packages: {', '.join(missing_packages)}")
        st.code(f"pip install {' '.join(missing_packages)}")
        st.stop()

# Check dependencies first
check_dependencies()

# Now import the packages
try:
    from sentence_transformers import SentenceTransformer, util
    import google.generativeai as genai
    from pypdf import PdfReader
    import torch
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="SmartNoteRAG",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'embedder' not in st.session_state:
    st.session_state.embedder = None

# Load embedding model with error handling
@st.cache_resource
def load_embedding_model():
    try:
        with st.spinner("Loading embedding model... This may take a moment on first run."):
            return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# Your existing functions with error handling
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_more_than_one_pdf(pdf_paths):
    all_text = ""
    for path in pdf_paths:
        try:
            pdf_text = extract_text_from_pdf(path)
            if pdf_text:
                filename = os.path.basename(path)
                all_text += f"\n\n### Content from: {filename} ###\n\n{pdf_text}\n"
        except Exception as e:
            st.warning(f"Error processing {path}: {e}")
            continue
    return all_text

def split_text(text, chunk_size=500, overlap=100):
    if not text:
        return []
    
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

def ask_pdf_question(pdf_paths, question, api_key, embedder, top_k=5):
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Extract text
        if isinstance(pdf_paths, list):
            pdf_text = extract_more_than_one_pdf(pdf_paths)
        else:
            pdf_text = extract_text_from_pdf(pdf_paths)
        
        if not pdf_text.strip():
            return "No text could be extracted from the PDF files.", []
        
        # Split into chunks
        chunks = split_text(pdf_text)
        
        if not chunks:
            return "No valid chunks could be created from the document.", []
        
        # Compute embeddings
        chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
        question_embedding = embedder.encode(question, convert_to_tensor=True)
        
        # Find most relevant chunks
        cosine_scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
        top_indices = cosine_scores.argsort(descending=True)[:top_k]
        top_chunks = [chunks[i] for i in top_indices]
        
        # Build prompt
        context = "\n\n".join(top_chunks)
        prompt = f"""You are an intelligent assistant. Answer the question strictly based on the document context provided below.

Do not use any external knowledge.

If the answer is not in the context, respond with "The answer is not available in the provided document."

Be concise and clear.

Document Context:
{context}

Question:
{question}

Answer:"""
        
        response = model.generate_content(prompt)
        return response.text, top_chunks
        
    except Exception as e:
        return f"Error processing question: {str(e)}", []

# Main Streamlit app
def main():
    st.title("🤖 SmartNoteRAG")
    st.caption("Upload PDFs and ask questions about their content using RAG (Retrieval-Augmented Generation)")

    with st.expander("ℹ️ How It Works – Step-by-Step Explanation"):
        st.markdown("""
    ### 🧠 SmartNoteRAG: What Happens Behind the Scenes?
    
    1. **📄 Upload a PDF File**  
       You start by uploading a lecture or research PDF. We extract all readable text from it.
    
    2. **✂️ Chunking the Text**  
       The text is split into smaller, overlapping pieces (called *chunks*) so we can search and understand it better later.
    
    3. **🔤 Embedding the Chunks**  
       Each chunk is converted into a numeric representation (embedding) using a pre-trained `sentence-transformers` model. This allows us to compare text semantically.
    
    4. **📚 Storing in a Vector Database (FAISS)**  
       All embeddings are stored in a FAISS index, a fast similarity search engine that helps retrieve the most relevant parts of the document later.
    
    5. **🔎 Retrieval of Relevant Chunks**  
       When you ask a question or request notes, the system finds the top document chunks most relevant to your query using vector similarity.
    
    6. **💬 Generation with Gemini**  
       The retrieved text chunks and your query are sent to **Gemini**, a powerful language model. It generates answers, summaries, or notes based only on the document content.
    
    7. **🎯 Final Output**  
       The result is a clean, concise response that’s grounded in the actual content of your uploaded PDF.
    
    > ⚠️ If your question can't be answered from the document, you'll be told so. No hallucinations here!
        """)

    # Load embedding model
    if st.session_state.embedder is None:
        st.session_state.embedder = load_embedding_model()
    
    if st.session_state.embedder is None:
        st.error("Failed to load embedding model. Please refresh the page.")
        st.stop()
    
    # Sidebar for upload and configuration
    with st.sidebar:
        st.header("📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            # Save uploaded files temporarily
            temp_paths = []
            try:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_paths.append(tmp_file.name)
                
                st.session_state.processed_files = temp_paths
                
                # Display uploaded files
                st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded successfully!")
                for file in uploaded_files:
                    st.write(f"📄 {file.name}")
                    
            except Exception as e:
                st.error(f"Error uploading files: {e}")
        
        st.divider()
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="Enter your Google AI API key to use Gemini model",
            placeholder="Enter your API key here..."
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            top_k = st.slider("Number of relevant chunks", 1, 10, 5)
            chunk_size = st.slider("Chunk size", 200, 1000, 500)
            overlap = st.slider("Chunk overlap", 0, 200, 100)
        
        # Example questions
        st.divider()
        st.header("💡 Example Questions")
        example_questions = [
            "What is the main topic of this document?",
            "Summarize the key points",
            "What are the main recommendations?",
            "Explain the methodology used"
        ]
        
        for q in example_questions:
            if st.button(q, use_container_width=True):
                st.session_state.example_question = q
        
        # Clear chat history
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat area
    st.divider()
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        # User question
        with st.chat_message("user"):
            st.write(chat["question"])
        
        # Assistant response
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            
            # Show relevant chunks
            
            #if chat.get("chunks"):
                #with st.expander("📋 View Relevant Document Chunks"):
                    #for j, chunk in enumerate(chat["chunks"]):
                        #st.caption(f"Chunk {j+1}")
                        #st.text(chunk)
                        #if j < len(chat["chunks"]) - 1:
                            #st.divider()
                            
    
    # Handle example question
    if hasattr(st.session_state, 'example_question'):
        question = st.session_state.example_question
        del st.session_state.example_question
    else:
        question = ""
    
    # Question input at bottom
    if prompt := st.chat_input("Ask a question about the PDFs..."):
        question = prompt
    
    if question:
        if not api_key:
            st.error("❌ Please enter your Google AI API key in the sidebar")
        elif not st.session_state.processed_files:
            st.error("❌ Please upload at least one PDF file")
        elif not question.strip():
            st.error("❌ Please enter a question")
        else:
            # Add user question to chat
            with st.chat_message("user"):
                st.write(question)
            
            # Process question
            with st.spinner("🔍 Analyzing documents..."):
                answer, relevant_chunks = ask_pdf_question(
                    st.session_state.processed_files,
                    question,
                    api_key,
                    st.session_state.embedder,
                    top_k
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "chunks": relevant_chunks
                })
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(answer)
                    
                    # Show relevant chunks
                    if relevant_chunks:
                        with st.expander("📋 View Relevant Document Chunks"):
                            for j, chunk in enumerate(relevant_chunks):
                                st.caption(f"Chunk {j+1}")
                                st.text(chunk)
                                if j < len(relevant_chunks) - 1:
                                    st.divider()
    
    # Footer
    st.divider()
    st.caption("Built by Ahmed Baalash using Streamlit, Sentence Transformers, and Google AI")

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.write("Please check the console for more details.")
