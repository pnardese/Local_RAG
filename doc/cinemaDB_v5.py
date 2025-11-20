import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions  # Added for custom embeddings
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import uuid
import re
import ollama
import streamlit as st

# --- REFINED CLASS DEFINITION ---
class DocumentToChromaDB:
    def __init__(self, documents_folder, db_path="./chroma_db", collection_name="documents"):
        """
        Initialize the Document to ChromaDB processor with Ollama Embeddings
        """
        self.documents_folder = documents_folder
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # --- REFINEMENT 1: Use Ollama for Embeddings ---
        # This uses "nomic-embed-text" which is much better than the default
        print("Initializing Ollama Embeddings (model: nomic-embed-text)...")
        self.ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url="http://localhost:11434/api/embeddings",
        )
        
        # Create or get collection with the custom embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ollama_ef,
            metadata={"description": "PDF and EPUB documents with page/chapter tracking"}
        )
        
        print(f"ChromaDB initialized at: {db_path}")
    
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
                    if not chapter_title:
                        chapter_title = f"Chapter {chapter_num}"
                    
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
            print(f"Error getting processed files: {e}")
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
                print(f"  Removed {len(ids_to_delete)} chunks from deleted file: {relative_path}")
        except Exception as e:
            print(f"Error removing {relative_path}: {e}")
    
    def process_documents(self, force_reprocess=False):
        """Process documents in the folder and subfolders"""
        if not os.path.exists(self.documents_folder):
            return f"Error: Folder {self.documents_folder} does not exist"
        
        all_docs = self.get_all_documents()
        
        if not all_docs:
            return f"No PDF or EPUB files found in {self.documents_folder}"
        
        processed_files = set() if force_reprocess else self.get_processed_files()
        
        if processed_files:
            current_doc_set = set(d['relative_path'] for d in all_docs)
            deleted_docs = processed_files - current_doc_set
            if deleted_docs:
                for deleted_doc in deleted_docs:
                    self.remove_document_from_db(deleted_doc)
        
        new_docs = [d for d in all_docs if d['relative_path'] not in processed_files]
        
        if not new_docs:
            return "All documents already processed."
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        # Create a progress bar for extracting text
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
                        all_metadatas.append({
                            "source": doc['filename'],
                            "relative_path": doc['relative_path'],
                            "file_type": doc['type'],
                            "page_number": page_data['page_number'],
                            "total_pages": page_data['total_pages'],
                            "chunk_id": chunk_idx,
                            "total_chunks_in_page": len(chunks)
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
                            "book_title": chapter_data['book_title'],
                            "chunk_id": chunk_idx,
                            "total_chunks_in_chapter": len(chunks)
                        })
                        all_ids.append(f"epub_{doc['filename']}_ch{chapter_data['chapter_number']}_c{chunk_idx}_{uuid.uuid4().hex[:8]}")
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # --- FIX: REDUCED BATCH SIZE FOR OLLAMA ---
        if all_documents:
            # Decreased from 2000 to 100 to prevent httpx.ReadTimeout
            BATCH_SIZE = 100
            total_chunks = len(all_documents)
            
            db_status = st.empty()
            db_progress = st.progress(0)
            
            print(f"Starting batch insertion of {total_chunks} chunks with batch size {BATCH_SIZE}...")
            
            for i in range(0, total_chunks, BATCH_SIZE):
                end_idx = min(i + BATCH_SIZE, total_chunks)
                
                # Update UI
                progress_percent = end_idx / total_chunks
                db_status.text(f"Generating Embeddings & Saving: {i} to {end_idx} of {total_chunks}...")
                db_progress.progress(progress_percent)
                
                batch_docs = all_documents[i:end_idx]
                batch_metas = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]
                
                try:
                    self.collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                except Exception as e:
                    print(f"Error adding batch {i}-{end_idx}: {e}")
                    return f"Error during database insertion at index {i}: {str(e)}"

            db_status.empty()
            db_progress.empty()
            
            return f"Successfully added {len(new_docs)} new document(s) ({total_chunks} chunks)."
        
        return "No valid text found in new documents."
  
    # --- REFINEMENT 2: Query Expansion ---
    def refine_query(self, query, llm_model):
        """Ask LLM to refine the query for better search retrieval"""
        prompt = f"""You are an AI search assistant. 
        Please rephrase the following user question to be more specific and suitable for a semantic vector search engine.
        If the query is already specific, just return it as is.
        Do not answer the question. Just provide the best search query string.
        
        User Question: {query}
        
        Search Query:"""
        
        try:
            response = ollama.chat(
                model=llm_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Query refinement failed: {e}")
            return query
        
    def search_and_answer(self, query, n_results=10, llm_model="llama3.2"):
        """Search documents and generate a complete answer using LLM"""
        
        # 1. Refine the query
        with st.status("Refining query...", expanded=False) as status:
            refined_query = self.refine_query(query, llm_model)
            status.update(label=f"Searching for: '{refined_query}'", state="complete", expanded=False)
        
        # 2. Search using the refined query
        results = self.collection.query(query_texts=[refined_query], n_results=n_results)
        
        if not results['documents'] or not results['documents'][0]:
            return {
                'answer': "No relevant documents found for your query.",
                'sources': []
            }
        
        context_parts = []
        sources = []
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), start=1):
            if metadata['file_type'] == 'pdf':
                source_info = f"[Source {i}: {metadata['source']}, Page {metadata['page_number']}]"
            else:
                source_info = f"[Source {i}: {metadata['source']}, Chapter {metadata['chapter_number']}: {metadata['chapter_title']}]"
            
            context_parts.append(f"{source_info}\n{doc}\n")
            sources.append({
                'source': metadata['relative_path'],
                'file_type': metadata['file_type'],
                'page_number': metadata.get('page_number'),
                'chapter_number': metadata.get('chapter_number'),
                'chapter_title': metadata.get('chapter_title'),
                'distance': results['distances'][0][i-1]
            })
        
        context = "\n".join(context_parts)
        
        # --- REFINEMENT 3: Stricter Prompt ---
        prompt = f"""You are an intelligent assistant analyzing the provided document excerpts.

STRICT INSTRUCTIONS:
1. Use ONLY the provided Context to answer the User Question.
2. If the answer is not in the Context, state "I cannot find the answer in the provided documents."
3. Cite your sources using the exact format [Source X] immediately after the relevant information.
4. Be concise but comprehensive.

User Question: {query}

Context:
{context}

Answer:"""
        
        try:
            response = ollama.chat(
                model=llm_model,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant that answers questions based on provided document excerpts. Always cite your sources.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return {
                'answer': response['message']['content'],
                'sources': sources,
                'raw_results': results
            }
        except Exception as e:
            return {
                'answer': f"Error: Could not generate answer. Make sure Ollama is running and model '{llm_model}' is available. Error: {str(e)}",
                'sources': sources,
                'raw_results': results
            }

# --- STREAMLIT INTERFACE ---

def main():
    st.set_page_config(page_title="CinemaDB RAG", page_icon="📚", layout="wide")
    
    st.title("📚 Local Document RAG Chat")
    
    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("Configuration")
        
        docs_folder = st.text_input("Documents Folder", "./documents")
        db_path = st.text_input("Database Path", "./chroma_db")
        llm_model = st.text_input("Ollama Model", "qwen3:4b") # Keeping your preferred model
        
        st.divider()
        
        st.info("💡 Note: If you are switching to the new version, please delete your old ./chroma_db folder and re-scan so the new embeddings take effect.")
        
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("Scan & Process", type="primary")
        with col2:
            reprocess_btn = st.button("Force Full Rescan")
            
        st.warning("Ensure 'ollama pull nomic-embed-text' has been run!")

    # --- INITIALIZE DB ---
    @st.cache_resource
    def get_db_instance(folder, path):
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        return DocumentToChromaDB(folder, path)

    doc_db = get_db_instance(docs_folder, db_path)

    # --- PROCESSING LOGIC ---
    if process_btn:
        with st.spinner("Scanning and processing documents..."):
            result_msg = doc_db.process_documents(force_reprocess=False)
            st.sidebar.success(result_msg)

    if reprocess_btn:
        with st.spinner("Force reprocessing all documents..."):
            result_msg = doc_db.process_documents(force_reprocess=True)
            st.sidebar.success(result_msg)

    # --- CHAT INTERFACE ---
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        loc = f"Page {src['page_number']}" if src['file_type'] == 'pdf' else f"Ch {src['chapter_number']}: {src['chapter_title']}"
                        st.markdown(f"**{i}. {src['source']}** ({loc})")

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                # --- REFINEMENT 4: Increased n_results ---
                # Increased from 4 to 8 for better context window
                result = doc_db.search_and_answer(prompt, n_results=8, llm_model=llm_model)
                
                answer = result['answer']
                sources = result['sources']
                
                message_placeholder.markdown(answer)
                
                if sources:
                    with st.expander("📚 Sources Used"):
                        for i, src in enumerate(sources, 1):
                            loc = f"Page {src['page_number']}" if src['file_type'] == 'pdf' else f"Ch {src['chapter_number']}: {src['chapter_title']}"
                            rel = f"{1 - src['distance']:.2%}"
                            st.markdown(f"**{i}. {src['source']}**\n* Location: {loc}\n* Relevance: {rel}")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })

if __name__ == "__main__":
    main()