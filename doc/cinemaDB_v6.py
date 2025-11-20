import os
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

# --- CLASS DEFINITION ---
class DocumentToChromaDB:
    def __init__(self, documents_folder, db_path, collection_name, embedding_model_name):
        """
        Initialize the Document to ChromaDB processor
        Args:
            embedding_model_name: The Ollama model used strictly for creating vectors (e.g., nomic-embed-text)
        """
        self.documents_folder = documents_folder
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        print(f"Initializing with Embedding Model: {self.embedding_model_name}")
        
        # Define the embedding function (The "Search" Model)
        self.ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            model_name=self.embedding_model_name,
            url="http://localhost:11434/api/embeddings",
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ollama_ef,
            metadata={"description": "PDF/EPUB docs", "embedding_model": self.embedding_model_name}
        )
    
    def get_all_documents(self):
        """Recursively get all PDF and EPUB files from the folder"""
        supported_extensions = {'.pdf', '.epub'}
        documents = []
        
        for root, dirs, files in os.walk(self.documents_folder):
            for file in files:
                if os.path.splitext(file.lower())[1] in supported_extensions:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.documents_folder)
                    documents.append({
                        'filename': file,
                        'full_path': full_path,
                        'relative_path': relative_path,
                        'type': os.path.splitext(file.lower())[1][1:]
                    })
        return documents
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with page numbers"""
        try:
            reader = PdfReader(pdf_path)
            pages_data = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    pages_data.append({
                        'text': text.strip(),
                        'page_number': page_num,
                        'total_pages': len(reader.pages)
                    })
            return pages_data
        except Exception as e:
            print(f"    Error reading PDF {pdf_path}: {e}")
            return None
    
    def extract_chapter_title(self, soup):
        """Try to extract chapter title from HTML content"""
        for tag in ['h1', 'h2', 'h3', 'title']:
            element = soup.find(tag)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) < 200:
                    return title
        return None
    
    def extract_text_from_epub(self, epub_path):
        """Extract text from EPUB with chapter information"""
        try:
            book = epub.read_epub(epub_path)
            chapters_data = []
            chapter_num = 0
            
            book_title = book.get_metadata('DC', 'title')
            book_title = book_title[0][0] if book_title else "Unknown"
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_num += 1
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    chapter_title = self.extract_chapter_title(soup)
                    if not chapter_title: chapter_title = f"Chapter {chapter_num}"
                    
                    text = soup.get_text(separator='\n', strip=True)
                    if text.strip():
                        chapters_data.append({
                            'text': text.strip(),
                            'chapter_number': chapter_num,
                            'chapter_title': chapter_title,
                            'book_title': book_title
                        })
            return chapters_data
        except Exception as e:
            print(f"    Error reading EPUB {epub_path}: {e}")
            return None
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into chunks for better semantic search"""
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
        """Get list of already processed files from metadata"""
        try:
            all_metadata = self.collection.get()
            if all_metadata and all_metadata['metadatas']:
                processed = set(meta['relative_path'] for meta in all_metadata['metadatas'])
                return processed
            return set()
        except Exception as e:
            return set()
    
    def remove_document_from_db(self, relative_path):
        """Remove all chunks of a specific document from the database"""
        try:
            all_data = self.collection.get()
            ids_to_delete = [
                id for id, meta in zip(all_data['ids'], all_data['metadatas'])
                if meta['relative_path'] == relative_path
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
        except Exception as e:
            print(f"Error removing {relative_path}: {e}")
    
    def process_documents(self, force_reprocess=False):
        """Process documents with smaller batch sizes to prevent timeouts"""
        if not os.path.exists(self.documents_folder):
            return f"Error: Folder {self.documents_folder} does not exist"
        
        all_docs = self.get_all_documents()
        if not all_docs: return "No documents found."
        
        processed_files = set() if force_reprocess else self.get_processed_files()
        
        # Handle deletions
        if processed_files:
            current_doc_set = set(d['relative_path'] for d in all_docs)
            deleted_docs = processed_files - current_doc_set
            for deleted_doc in deleted_docs:
                self.remove_document_from_db(deleted_doc)
        
        new_docs = [d for d in all_docs if d['relative_path'] not in processed_files]
        if not new_docs: return "All documents already processed."
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Extract Text
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
                        all_metadatas.append({
                            "source": doc['filename'],
                            "relative_path": doc['relative_path'],
                            "file_type": doc['type'],
                            "page_number": page_data['page_number'],
                            "chunk_id": chunk_idx,
                        })
                        all_ids.append(f"pdf_{doc['filename']}_p{page_data['page_number']}_c{chunk_idx}_{uuid.uuid4().hex[:8]}")
            
            elif doc['type'] == 'epub':
                chapters_data = self.extract_text_from_epub(doc['full_path'])
                if not chapters_data: continue
                for chapter_data in chapters_data:
                    chunks = self.chunk_text(chapter_data['text'])
                    for chunk_idx, chunk in enumerate(chunks):
                        all_documents.append(chunk)
                        all_metadatas.append({
                            "source": doc['filename'],
                            "relative_path": doc['relative_path'],
                            "file_type": doc['type'],
                            "chapter_number": chapter_data['chapter_number'],
                            "chapter_title": chapter_data['chapter_title'],
                            "chunk_id": chunk_idx,
                        })
                        all_ids.append(f"epub_{doc['filename']}_ch{chapter_data['chapter_number']}_c{chunk_idx}_{uuid.uuid4().hex[:8]}")
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # 2. Embed and Insert (Batch Size = 100)
        if all_documents:
            BATCH_SIZE = 100 # Reduced to prevent httpx.ReadTimeout
            total_chunks = len(all_documents)
            
            db_status = st.empty()
            db_progress = st.progress(0)
            
            for i in range(0, total_chunks, BATCH_SIZE):
                end_idx = min(i + BATCH_SIZE, total_chunks)
                
                db_status.text(f"Generating Vectors ({self.embedding_model_name}): {i} to {end_idx} of {total_chunks}...")
                db_progress.progress(end_idx / total_chunks)
                
                batch_docs = all_documents[i:end_idx]
                batch_metas = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]
                
                try:
                    self.collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                except Exception as e:
                    return f"Error at index {i}: {str(e)}"

            db_status.empty()
            db_progress.empty()
            return f"Successfully added {len(new_docs)} new document(s) ({total_chunks} chunks)."
        
        return "No valid text found."

    def refine_query(self, query, chat_model):
        """Use the Chat Model to refine the search query"""
        prompt = f"""You are an AI search assistant. 
        Rephrase the following user question to be more specific for a semantic search engine.
        Do not answer the question. Just return the best query string.
        User Question: {query}"""
        
        try:
            response = ollama.chat(model=chat_model, messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content'].strip()
        except:
            return query
        
    def search_and_answer(self, query, chat_model, n_results=8):
        """
        1. Refine Query (using Chat Model)
        2. Search DB (using Embedding Model)
        3. Generate Answer (using Chat Model)
        """
        
        # 1. Refine
        with st.status(f"Refining query with {chat_model}...", expanded=False) as status:
            refined_query = self.refine_query(query, chat_model)
            status.update(label=f"Searching for: '{refined_query}'", state="complete")
        
        # 2. Search (Chroma uses the embedding model defined in __init__ automatically)
        results = self.collection.query(query_texts=[refined_query], n_results=n_results)
        
        if not results['documents'] or not results['documents'][0]:
            return {'answer': "No documents found.", 'sources': []}
        
        # 3. Prepare Context
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
        
        # 4. Generate Answer
        prompt = f"""Use the provided Context to answer the User Question.
        If the answer is not in the context, say so.
        Cite sources as [Source X].

        User Question: {query}
        
        Context:
        {"\n".join(context_parts)}
        """
        
        try:
            response = ollama.chat(model=chat_model, messages=[{'role': 'user', 'content': prompt}])
            return {'answer': response['message']['content'], 'sources': sources}
        except Exception as e:
            return {'answer': f"Error generating answer: {e}", 'sources': sources}

# --- STREAMLIT INTERFACE ---
def main():
    st.set_page_config(page_title="CinemaDB RAG", page_icon="📚", layout="wide")
    st.title("📚 Two-Model Local RAG")
    
    with st.sidebar:
        st.header("AI Model Configuration")
        
        # --- TWO DIFFERENT MODELS INPUT ---
        embedding_model = st.text_input("1. Embedding Model (Search)", "nomic-embed-text", help="Used for indexing files. If you change this, you MUST Force Full Rescan.")
        chat_model = st.text_input("2. Chat Model (Answer)", "llama3.2", help="Used for talking to you.")
        
        st.divider()
        st.header("Folders")
        docs_folder = st.text_input("Documents Folder", "./documents")
        db_path = st.text_input("Database Path", "./chroma_db")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("Update New Files", type="primary")
        with col2:
            reprocess_btn = st.button("Force Full Rescan", help="Click this if you changed the Embedding Model")

    # --- INITIALIZE DB (Passes embedding model) ---
    @st.cache_resource(show_spinner=False)
    def get_db_instance(folder, path, embed_model):
        if not os.path.exists(folder): os.makedirs(folder, exist_ok=True)
        return DocumentToChromaDB(folder, path, "documents", embed_model)

    # We pass the embedding_model here so the cache invalidates if you change it
    doc_db = get_db_instance(docs_folder, db_path, embedding_model)

    if process_btn:
        with st.spinner("Processing..."):
            st.sidebar.success(doc_db.process_documents(force_reprocess=False))

    if reprocess_btn:
        with st.spinner("Re-indexing everything..."):
            st.sidebar.success(doc_db.process_documents(force_reprocess=True))

    # --- CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {src['source']}**")

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # Pass the CHAT MODEL here
            result = doc_db.search_and_answer(prompt, chat_model=chat_model, n_results=8)
            st.markdown(result['answer'])
            if result['sources']:
                with st.expander("📚 Sources Used"):
                    for i, src in enumerate(result['sources'], 1):
                        rel = f"{1 - src['distance']:.2%}"
                        st.markdown(f"**{i}. {src['source']}** (Relevance: {rel})")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": result['answer'],
            "sources": result['sources']
        })

if __name__ == "__main__":
    main()