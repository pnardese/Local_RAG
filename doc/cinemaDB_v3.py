import os
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import uuid
import re
import ollama


class DocumentToChromaDB:
    def __init__(self, documents_folder, db_path="./chroma_db", collection_name="documents"):
        """
        Initialize the Document to ChromaDB processor
        
        Args:
            documents_folder: Path to folder containing PDF and EPUB files (searches recursively)
            db_path: Path where ChromaDB will store data
            collection_name: Name of the ChromaDB collection
        """
        self.documents_folder = documents_folder
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF and EPUB documents with page/chapter tracking"}
        )
        
        print(f"ChromaDB initialized at: {db_path}")
        print(f"Collection: {collection_name}")
    
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
        # Look for common chapter title patterns
        for tag in ['h1', 'h2', 'h3', 'title']:
            element = soup.find(tag)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) < 200:  # Reasonable title length
                    return title
        return None
    
    def extract_text_from_epub(self, epub_path):
        """Extract text from EPUB with chapter information"""
        try:
            book = epub.read_epub(epub_path)
            chapters_data = []
            chapter_num = 0
            
            # Try to get book metadata
            book_title = book.get_metadata('DC', 'title')
            book_title = book_title[0][0] if book_title else "Unknown"
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_num += 1
                    
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    
                    # Try to extract chapter title
                    chapter_title = self.extract_chapter_title(soup)
                    if not chapter_title:
                        chapter_title = f"Chapter {chapter_num}"
                    
                    # Extract text
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
        """
        Split text into chunks for better semantic search
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
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
        """
        Process documents in the folder and subfolders, add new ones to ChromaDB
        
        Args:
            force_reprocess: If True, reprocess all documents even if already in DB
        """
        if not os.path.exists(self.documents_folder):
            print(f"Error: Folder {self.documents_folder} does not exist")
            return
        
        # Get all documents recursively
        all_docs = self.get_all_documents()
        
        if not all_docs:
            print(f"No PDF or EPUB files found in {self.documents_folder}")
            return
        
        print(f"\nFound {len(all_docs)} document(s) in folder (including subfolders)")
        pdf_count = sum(1 for d in all_docs if d['type'] == 'pdf')
        epub_count = sum(1 for d in all_docs if d['type'] == 'epub')
        print(f"  PDFs: {pdf_count}, EPUBs: {epub_count}")
        
        # Get already processed files
        processed_files = set() if force_reprocess else self.get_processed_files()
        
        # Check for deleted documents and remove from DB
        if processed_files:
            current_doc_set = set(d['relative_path'] for d in all_docs)
            deleted_docs = processed_files - current_doc_set
            if deleted_docs:
                print(f"\nDetected {len(deleted_docs)} deleted document(s), removing from database...")
                for deleted_doc in deleted_docs:
                    self.remove_document_from_db(deleted_doc)
        
        # Filter to only new documents
        new_docs = [d for d in all_docs if d['relative_path'] not in processed_files]
        
        if not new_docs:
            print("✓ All documents already processed. Database is up to date.")
            return 0
        
        print(f"\nProcessing {len(new_docs)} new document(s)...")
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for doc in new_docs:
            print(f"\n  Processing: {doc['relative_path']} ({doc['type'].upper()})")
            
            # Extract text based on file type
            if doc['type'] == 'pdf':
                pages_data = self.extract_text_from_pdf(doc['full_path'])
                if not pages_data:
                    continue
                
                # Process each page
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
                
                print(f"    Processed {len(pages_data)} pages")
            
            elif doc['type'] == 'epub':
                chapters_data = self.extract_text_from_epub(doc['full_path'])
                if not chapters_data:
                    continue
                
                # Process each chapter
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
                
                print(f"    Processed {len(chapters_data)} chapters")
        
        # Add all documents to ChromaDB
        if all_documents:
            print(f"\n  Adding {len(all_documents)} chunks to ChromaDB...")
            self.collection.add(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"✓ Successfully added {len(new_docs)} new document(s) to ChromaDB")
        
        return len(all_documents)
    
    def search(self, query, n_results=5):
        """
        Perform semantic search on the collection
        
        Args:
            query: Search query string
            n_results: Number of results to return
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return results
    
    def search_and_answer(self, query, n_results=5, llm_model="llama3.2"):
        """
        Search documents and generate a complete answer using LLM
        
        Args:
            query: Search query string
            n_results: Number of search results to use as context
            llm_model: Ollama model to use for answer generation
        """
        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return {
                'answer': "No relevant documents found for your query.",
                'sources': []
            }
        
        # Build context from search results
        context_parts = []
        sources = []
        
        for i, (doc, metadata) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
        ), start=1):
            # Format source information
            if metadata['file_type'] == 'pdf':
                source_info = f"[Source {i}: {metadata['source']}, Page {metadata['page_number']}]"
            else:  # epub
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
        
        # Create prompt for LLM
        prompt = f"""Based on the following excerpts from documents, please answer this question: {query}

Context from documents:
{context}

Please provide a comprehensive answer based on the information above. If the context doesn't contain enough information to fully answer the question, please say so. Always cite which sources you're using (e.g., "According to Source 1..." or "As mentioned in Source 2...").

Answer:"""
        
        # Generate answer using Ollama
        try:
            print(f"\n🤖 Generating answer using {llm_model}...")
            response = ollama.chat(
                model=llm_model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant that answers questions based on provided document excerpts. Always cite your sources when answering.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            answer = response['message']['content']
            
            return {
                'answer': answer,
                'sources': sources,
                'raw_results': results
            }
            
        except Exception as e:
            print(f"Error generating answer with LLM: {e}")
            return {
                'answer': f"Error: Could not generate answer. Make sure Ollama is running and model '{llm_model}' is available.",
                'sources': sources,
                'raw_results': results
            }


def main():
    # Configuration
    DOCUMENTS_FOLDER = "./documents"
    DB_PATH = "./chroma_db"
    COLLECTION_NAME = "documents"
    LLM_MODEL = "llama3.2"  # Or: llama3.1, mistral, phi3, etc.
    
    # Initialize and process documents
    doc_db = DocumentToChromaDB(DOCUMENTS_FOLDER, DB_PATH, COLLECTION_NAME)
    num_chunks = doc_db.process_documents()
    
    if num_chunks > 0:
        print("\n" + "="*50)
        print("Database created successfully!")
        print("="*50)
    
    # Example: Traditional search (just retrieval)
    print("\n" + "="*50)
    print("EXAMPLE 1: Traditional Search")
    print("="*50)
    query = "What is the main topic?"
    results = doc_db.search(query, n_results=3)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results['documents'][0])} results:")
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {metadata['relative_path']}")
        print(f"Type: {metadata['file_type'].upper()}")
        
        if metadata['file_type'] == 'pdf':
            print(f"Page: {metadata['page_number']}/{metadata['total_pages']}")
        elif metadata['file_type'] == 'epub':
            print(f"Chapter {metadata['chapter_number']}: {metadata['chapter_title']}")
        
        print(f"Relevance: {1 - distance:.4f}")
        print(f"Text preview: {doc[:150]}...")
    
    # Example: RAG - Search and generate complete answer
    print("\n" + "="*50)
    print("EXAMPLE 2: RAG (Retrieval-Augmented Generation)")
    print("="*50)
    query = "Provide a list of the main software tools mentioned."
    result = doc_db.search_and_answer(query, n_results=5, llm_model=LLM_MODEL)
    
    print(f"\nQuery: {query}\n")
    print("Answer:")
    print("-" * 50)
    print(result['answer'])
    print("-" * 50)
    
    print("\n📚 Sources used:")
    for i, source in enumerate(result['sources'], start=1):
        print(f"\n{i}. {source['source']}")
        if source['file_type'] == 'pdf':
            print(f"   Page {source['page_number']}")
        else:
            print(f"   Chapter {source['chapter_number']}: {source['chapter_title']}")
        print(f"   Relevance: {1 - source['distance']:.4f}")


if __name__ == "__main__":
    main()