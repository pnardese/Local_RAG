import os
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import uuid

class PDFToChromaDB:
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
            metadata={"description": "PDF and EPUB documents for semantic search"}
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
                    # Store relative path for better organization
                    relative_path = os.path.relpath(full_path, self.documents_folder)
                    documents.append({
                        'filename': file,
                        'full_path': full_path,
                        'relative_path': relative_path,
                        'type': os.path.splitext(file.lower())[1][1:]  # Remove the dot
                    })
        
        return documents
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"    Error reading PDF {pdf_path}: {e}")
            return None
    
    def extract_text_from_epub(self, epub_path):
        """Extract text from an EPUB file"""
        try:
            book = epub.read_epub(epub_path)
            text = ""
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    # Extract text and add newlines between paragraphs
                    text += soup.get_text(separator='\n', strip=True) + "\n\n"
            
            return text.strip()
        except Exception as e:
            print(f"    Error reading EPUB {epub_path}: {e}")
            return None
    
    def extract_text(self, file_path, file_type):
        """Extract text based on file type"""
        if file_type == 'pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_type == 'epub':
            return self.extract_text_from_epub(file_path)
        else:
            print(f"    Unsupported file type: {file_type}")
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
                
                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
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
    
    def process_pdfs(self, force_reprocess=False):
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
            text = self.extract_text(doc['full_path'], doc['type'])
            if not text:
                continue
            
            # Chunk the text
            chunks = self.chunk_text(text)
            print(f"    Created {len(chunks)} chunks")
            
            # Add chunks to collection data
            for i, chunk in enumerate(chunks):
                all_documents.append(chunk)
                all_metadatas.append({
                    "source": doc['filename'],
                    "relative_path": doc['relative_path'],
                    "file_type": doc['type'],
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
                all_ids.append(f"{doc['type']}_{doc['filename']}_{i}_{uuid.uuid4().hex[:8]}")
        
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


def main():
    # Configuration
    DOCUMENTS_FOLDER = "./documents"  # Change this to your documents folder path
    DB_PATH = "./chroma_db"
    COLLECTION_NAME = "documents"
    
    # Initialize and process documents
    pdf_db = PDFToChromaDB(DOCUMENTS_FOLDER, DB_PATH, COLLECTION_NAME)
    num_chunks = pdf_db.process_pdfs()
    
    if num_chunks > 0:
        print("\n" + "="*50)
        print("Database created successfully!")
        print("="*50)
        
        # Example search
        print("\nExample search:")
        query = "What is the main topic?"  # Change this to test search
        results = pdf_db.search(query, n_results=3)
        
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
            print(f"Chunk: {metadata['chunk_id'] + 1}/{metadata['total_chunks']}")
            print(f"Distance: {distance:.4f}")
            print(f"Text preview: {doc[:200]}...")


if __name__ == "__main__":
    main()