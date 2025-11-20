import os
import shutil
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import uuid
import re
import ollama
import streamlit as st
import time

# --- EXTERNAL API IMPORT ---
# NOTE: You MUST install this: pip install openai
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("Install 'openai' library to use external models (pip install openai).")

# --- HELPER FUNCTION: Fetch Ollama Models ---
@st.cache_data(ttl=300) 
def get_ollama_models():
    """Fetches a list of available models from the local Ollama server."""
    try:
        models = ollama.list()
        if 'models' not in models:
            return [], "Ollama server connected, but model list format is unexpected."
        
        model_names = [m['name'] for m in models['models']]
        
        if not model_names:
            return [], "Ollama server running, but no models found. Run 'ollama pull <model_name>'."

        return model_names, None

    except ConnectionError:
        error_msg = "Could not connect to Ollama. Ensure the server is running."
        return [], error_msg
        
    except Exception as e:
        error_msg = f"Failed to retrieve models. Server may be offline or returned unexpected data: {e}"
        return [], error_msg

# --- CLASS DEFINITION ---
class DocumentToChromaDB:
    def __init__(self, documents_folder, db_path, collection_name, embedding_model_name):
        # ... (Initialization logic is the same as V8)
        self.documents_folder = documents_folder
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            model_name=self.embedding_model_name,
            url="http://localhost:11434/api/embeddings",
        )
        
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.ollama_ef,
                metadata={"description": "PDF/EPUB docs", "embedding_model": self.embedding_model_name}
            )
        except Exception as e:
            st.error(f"Error initializing collection: {e}. If you changed models, click 'Reset Database'.")

    def reset_database(self):
        # ... (Same as V8)
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.ollama_ef,
                metadata={"description": "PDF/EPUB docs", "embedding_model": self.embedding_model_name}
            )
            return True, "Database deleted and re-initialized. Ready for new model."
        except Exception as e:
            return False, f"Error resetting database: {e}"
            
    # --- (Other helper methods like get_all_documents, extract_text_from_pdf, chunk_text are omitted for brevity) ---
    def get_all_documents(self):
        supported_extensions = {'.pdf', '.epub'}
        documents = []
        for root, dirs, files in os.walk(self.documents_folder):
            for file in files:
                if os.path.splitext(file.lower())[1] in supported_extensions:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.documents_folder)
                    documents.append({'filename': file, 'full_path': full_path, 'relative_path': relative_path, 'type': os.path.splitext(file.lower())[1][1:]})
        return documents

    def extract_text_from_pdf(self, pdf_path):
        try:
            reader = PdfReader(pdf_path)
            pages_data = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    pages_data.append({'text': text.strip(), 'page_number': page_num})
            return pages_data
        except Exception as e:
            return None
    
    def extract_text_from_epub(self, epub_path):
        try:
            book = epub.read_epub(epub_path)
            chapters_data = []
            chapter_num = 0
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_num += 1
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    if text.strip():
                        chapters_data.append({'text': text.strip(), 'chapter_number': chapter_num, 'chapter_title': f"Chapter {chapter_num}"})
            return chapters_data
        except Exception as e:
            return None
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size * 0.5:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            chunks.append(chunk.strip())
            start = end - overlap
        return chunks
    
    def get_processed_files(self):
        try:
            all_metadata = self.collection.get()
            if all_metadata and all_metadata['metadatas']:
                return set(meta['relative_path'] for meta in all_metadata['metadatas'])
            return set()
        except:
            return set()
    
    def process_documents(self, force_reprocess=False):
        if not os.path.exists(self.documents_folder):
            return f"Error: Folder {self.documents_folder} does not exist"
        
        all_docs = self.get_all_documents()
        if not all_docs: return "No documents found."
        processed_files = set() if force_reprocess else self.get_processed_files()
        new_docs = [d for d in all_docs if d['relative_path'] not in processed_files]
        if not new_docs: return "All documents already processed."
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, doc in enumerate(new_docs):
            status_text.text(f"Reading file: {doc['relative_path']}")
            progress_bar.progress((idx) / len(new_docs))
            
            if doc['type'] == 'pdf':
                pages_data = self.extract_text_from_pdf(doc['full_path'])
                if not pages_data: continue
                for page_data in pages_data:
                    chunks = self.chunk_text(page_data['text'])
                    for chunk_idx, chunk in enumerate(chunks):
                        all_documents.append(chunk)
                        all_metadatas.append({"source": doc['filename'], "relative_path": doc['relative_path'], "file_type": doc['type'], "page_number": page_data['page_number'], "chunk_id": chunk_idx})
                        all_ids.append(f"pdf_{doc['filename']}_p{page_data['page_number']}_c{chunk_idx}_{uuid.uuid4().hex[:8]}")
            
            elif doc['type'] == 'epub':
                chapters_data = self.extract_text_from_epub(doc['full_path'])
                if not chapters_data: continue
                for chapter_data in chapters_data:
                    chunks = self.chunk_text(chapter_data['text'])
                    for chunk_idx, chunk in enumerate(chunks):
                        all_documents.append(chunk)
                        all_metadatas.append({"source": doc['filename'], "relative_path": doc['relative_path'], "file_type": doc['type'], "chapter_number": chapter_data['chapter_number'], "chapter_title": chapter_data['chapter_title'], "chunk_id": chunk_idx})
                        all_ids.append(f"epub_{doc['filename']}_ch{chapter_data['chapter_number']}_c{chunk_idx}_{uuid.uuid4().hex[:8]}")
        
        progress_bar.progress(1.0)
        status_text.empty()

        if all_documents:
            BATCH_SIZE = 100
            total_chunks = len(all_documents)
            db_status = st.empty()
            db_progress = st.progress(0)
            
            for i in range(0, total_chunks, BATCH_SIZE):
                end_idx = min(i + BATCH_SIZE, total_chunks)
                db_status.text(f"Embedding & Saving: {i} to {end_idx} of {total_chunks}...")
                db_progress.progress(end_idx / total_chunks)
                
                try:
                    self.collection.add(
                        documents=all_documents[i:end_idx], 
                        metadatas=all_metadatas[i:end_idx], 
                        ids=all_ids[i:end_idx]
                    )
                except Exception as e:
                    return f"Error at index {i}: {str(e)}"

            db_status.empty()
            db_progress.empty()
            return f"Successfully added {len(new_docs)} documents ({total_chunks} chunks)."
        return "No valid text found."

    def search_and_answer(self, query, provider, chat_model, n_results, api_key=None):
        # 1. Search
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
        except Exception as e:
             return {'answer': f"Search Error: {e}. \n\n**Tip:** If you changed models, click 'Reset Database'.", 'sources': []}

        if not results['documents'] or not results['documents'][0]:
            return {'answer': "No documents found.", 'sources': []}
        
        # 2. Prepare Context
        context_parts = []
        sources = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), start=1):
            loc = f"Page {metadata.get('page_number')}" if metadata['file_type'] == 'pdf' else f"Ch {metadata.get('chapter_number')}"
            context_parts.append(f"[Source {i}: {metadata['source']}, {loc}]\n{doc}\n")
            sources.append({
                'source': metadata['relative_path'],
                'file_type': metadata['file_type'],
                'page_number': metadata.get('page_number'),
                'chapter_number': metadata.get('chapter_number'),
                'chapter_title': metadata.get('chapter_title'),
                'distance': results['distances'][0][i-1]
            })
        
        # 3. Generate Answer
        system_prompt = f"""You are a helpful assistant. Use the provided Context to answer the User Question.
        If the answer is not in the context, state that you could not find the information in the documents.
        Cite sources as [Source X].
        
        User Question: {query}
        
        Context:
        {"\n".join(context_parts)}"""
        
        messages = [{'role': 'user', 'content': system_prompt}]
        
        try:
            if provider == "Ollama (Local)":
                # Use Ollama
                response = ollama.chat(model=chat_model, messages=messages)
                answer = response['message']['content']
            
            elif provider == "External API (OpenAI)":
                if not OPENAI_AVAILABLE:
                    return {'answer': "External API selected, but the 'openai' library is not installed.", 'sources': sources}
                
                if not api_key:
                    return {'answer': "External API selected, but no API key was provided.", 'sources': sources}

                # Use OpenAI
                client = OpenAI(api_key=api_key)
                
                response = client.chat.completions.create(
                    model=chat_model,
                    messages=messages,
                    temperature=0.0 # Set low for RAG consistency
                )
                answer = response.choices[0].message.content
            
            else:
                 answer = "Error: Invalid model provider selected."

            return {'answer': answer, 'sources': sources}
        
        except Exception as e:
            return {'answer': f"LLM Error ({provider}): {e}", 'sources': sources}

# --- STREAMLIT INTERFACE ---
def main():
    st.set_page_config(page_title="Hybrid RAG", page_icon="📚", layout="wide")
    st.title("📚 Hybrid RAG Manager (Ollama/OpenAI)")
    
    available_models, model_fetch_error = get_ollama_models()
    
    DEFAULT_EMBED = "mxbai-embed-large"
    DEFAULT_CHAT = "llama3.2"
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("Configuration")
        
        # 1. Model Provider Selection
        provider = st.radio(
            "Select LLM Provider:",
            options=("Ollama (Local)", "External API (OpenAI)"),
            index=0 # Default to Ollama
        )
        st.divider()

        # 2. Embedding Model (Always Ollama)
        st.subheader("Embedding (Indexing) Model")
        if model_fetch_error:
            st.error(model_fetch_error)
            embedding_model = st.text_input("Embedding Model (Manual)", DEFAULT_EMBED)
        else:
            def get_default_index(model_list, default_name):
                try: return model_list.index(default_name)
                except ValueError: return 0
            
            embedding_model = st.selectbox(
                "Embedding Model (Search)", 
                options=available_models,
                index=get_default_index(available_models, DEFAULT_EMBED)
            )
        
        st.divider()
        
        # 3. Chat Model (Conditional Inputs)
        st.subheader("Chat (Answering) Model")
        if provider == "Ollama (Local)":
            if model_fetch_error:
                 chat_model = st.text_input("Chat Model (Manual)", DEFAULT_CHAT)
            else:
                 chat_model = st.selectbox(
                    "Chat Model (Answer)", 
                    options=available_models,
                    index=get_default_index(available_models, DEFAULT_CHAT)
                )

        elif provider == "External API (OpenAI)":
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Get your key from platform.openai.com. It is not saved.",
            )
            chat_model = st.text_input("OpenAI Model Name", DEFAULT_OPENAI_MODEL, help="e.g., gpt-4o, gpt-3.5-turbo")
            
            if not api_key:
                 st.warning("API Key is required for External API use.")
            
        else:
            chat_model = DEFAULT_CHAT
            api_key = None


        # 4. Search & Folder Config (Same as V8)
        st.divider()
        num_results = st.slider("Sources to Retrieve (n_results)", 1, 20, 8)
        docs_folder = st.text_input("Docs Folder", "./documents")
        db_path = st.text_input("DB Path", "./chroma_db")
        
        st.divider()
        if st.button("Scan & Process New Files", type="primary"):
            db = DocumentToChromaDB(docs_folder, db_path, "documents", embedding_model)
            with st.spinner("Processing..."):
                st.success(db.process_documents())

        st.divider()
        st.markdown("### ⚠️ Danger Zone")
        if st.button("Reset Database (Clear All)"):
            db = DocumentToChromaDB(docs_folder, db_path, "documents", embedding_model)
            success, msg = db.reset_database()
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    
    # --- CHAT AREA ---
    if "messages" not in st.session_state: st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Sources"):
                    for s in msg["sources"]: st.markdown(f"- {s['source']}")

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            db = DocumentToChromaDB(docs_folder, db_path, "documents", embedding_model)
            
            # --- Pass all required parameters ---
            result = db.search_and_answer(
                query=prompt, 
                provider=provider, 
                chat_model=chat_model, 
                n_results=num_results,
                api_key=st.session_state.get('openai_api_key', api_key) # Use key if available
            )
            
            st.markdown(result['answer'])
            if result['sources']:
                with st.expander("Sources Used"):
                    for i, src in enumerate(result['sources'], 1):
                         st.markdown(f"**{i}. {src['source']}** ({1-src['distance']:.0%})")
        
        st.session_state.messages.append({"role": "assistant", "content": result['answer'], "sources": result['sources']})

if __name__ == "__main__":
    # Workaround to pass API key to search_and_answer without a re-run
    # Not strictly necessary but prevents loss if the user clicks other buttons
    if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = None
    
    # Run the main function
    main()