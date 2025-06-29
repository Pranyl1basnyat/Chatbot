import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Uncomment the following lines if you're not using pipenv as your virtual environment manager
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

class DocumentProcessor:
    """Enhanced document processing pipeline for PDF files"""
    
    def __init__(self, 
                 data_path: str = "data/",
                 db_path: str = "vectorstore/db_faiss",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the document processor
        
        Args:
            data_path: Path to directory containing PDF files
            db_path: Path to save FAISS vector database
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            embedding_model_name: Name of the embedding model to use
        """
        self.data_path = Path(data_path)
        self.db_path = Path(db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        
        # Create directories if they don't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = None
        
    def load_pdf_files(self, verbose: bool = True) -> List[Document]:
        """
        Load PDF files from the data directory
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            List of loaded documents
        """
        try:
            if verbose:
                print(f"Loading PDF files from: {self.data_path}")
            
            # Check if data directory exists and has PDF files
            pdf_files = list(self.data_path.glob("*.pdf"))
            if not pdf_files:
                print(f"No PDF files found in {self.data_path}")
                return []
            
            if verbose:
                print(f"Found {len(pdf_files)} PDF files:")
                for pdf_file in pdf_files:
                    print(f"  - {pdf_file.name}")
            
            # Load documents using DirectoryLoader
            loader = DirectoryLoader(
                str(self.data_path),
                glob='*.pdf',
                loader_cls=PyPDFLoader,
                show_progress=verbose
            )
            
            documents = loader.load()
            
            if verbose:
                print(f"Successfully loaded {len(documents)} document pages")
                
            return documents
            
        except Exception as e:
            print(f"Error loading PDF files: {str(e)}")
            return []
    
    def create_chunks(self, documents: List[Document], verbose: bool = True) -> List[Document]:
        """
        Create text chunks from documents
        
        Args:
            documents: List of documents to chunk
            verbose: Whether to print progress information
            
        Returns:
            List of text chunks
        """
        try:
            if not documents:
                print("No documents provided for chunking")
                return []
            
            if verbose:
                print(f"Creating text chunks with size={self.chunk_size}, overlap={self.chunk_overlap}")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            text_chunks = text_splitter.split_documents(documents)
            
            if verbose:
                print(f"Created {len(text_chunks)} text chunks")
                
                # Show statistics
                chunk_lengths = [len(chunk.page_content) for chunk in text_chunks]
                avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
                print(f"Average chunk length: {avg_length:.0f} characters")
                print(f"Min chunk length: {min(chunk_lengths) if chunk_lengths else 0}")
                print(f"Max chunk length: {max(chunk_lengths) if chunk_lengths else 0}")
            
            return text_chunks
            
        except Exception as e:
            print(f"Error creating chunks: {str(e)}")
            return []
    
    def get_embedding_model(self, verbose: bool = True) -> Optional[HuggingFaceEmbeddings]:
        """
        Get or initialize the embedding model
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            HuggingFace embeddings model
        """
        try:
            if self.embedding_model is None:
                if verbose:
                    print(f"Loading embedding model: {self.embedding_model_name}")
                
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                if verbose:
                    print("Embedding model loaded successfully")
            
            return self.embedding_model
            
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            return None
    
    def create_vectorstore(self, text_chunks: List[Document], verbose: bool = True) -> Optional[FAISS]:
        """
        Create FAISS vectorstore from text chunks
        
        Args:
            text_chunks: List of text chunks to vectorize
            verbose: Whether to print progress information
            
        Returns:
            FAISS vectorstore
        """
        try:
            if not text_chunks:
                print("No text chunks provided for vectorstore creation")
                return None
            
            # Get embedding model
            embedding_model = self.get_embedding_model(verbose=verbose)
            if embedding_model is None:
                return None
            
            if verbose:
                print(f"Creating FAISS vectorstore from {len(text_chunks)} chunks...")
                start_time = time.time()
            
            # Create FAISS vectorstore
            vectorstore = FAISS.from_documents(text_chunks, embedding_model)
            
            if verbose:
                elapsed_time = time.time() - start_time
                print(f"Vectorstore created in {elapsed_time:.2f} seconds")
            
            return vectorstore
            
        except Exception as e:
            print(f"Error creating vectorstore: {str(e)}")
            return None
    
    def save_vectorstore(self, vectorstore: FAISS, verbose: bool = True) -> bool:
        """
        Save FAISS vectorstore to disk
        
        Args:
            vectorstore: FAISS vectorstore to save
            verbose: Whether to print progress information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if vectorstore is None:
                print("No vectorstore provided for saving")
                return False
            
            if verbose:
                print(f"Saving vectorstore to: {self.db_path}")
            
            vectorstore.save_local(str(self.db_path))
            
            if verbose:
                print("Vectorstore saved successfully")
                
            return True
            
        except Exception as e:
            print(f"Error saving vectorstore: {str(e)}")
            return False
    
    def load_vectorstore(self, verbose: bool = True) -> Optional[FAISS]:
        """
        Load FAISS vectorstore from disk
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Loaded FAISS vectorstore
        """
        try:
            if not self.db_path.exists():
                if verbose:
                    print(f"Vectorstore not found at: {self.db_path}")
                return None
            
            # Get embedding model
            embedding_model = self.get_embedding_model(verbose=verbose)
            if embedding_model is None:
                return None
            
            if verbose:
                print(f"Loading vectorstore from: {self.db_path}")
            
            vectorstore = FAISS.load_local(
                str(self.db_path), 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            
            if verbose:
                print("Vectorstore loaded successfully")
                
            return vectorstore
            
        except Exception as e:
            print(f"Error loading vectorstore: {str(e)}")
            return None
    
    def process_documents(self, verbose: bool = True) -> Tuple[bool, str]:
        """
        Complete document processing pipeline
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if verbose:
                print("="*50)
                print("Starting Document Processing Pipeline")
                print("="*50)
                start_time = time.time()
            
            # Step 1: Load PDF files
            documents = self.load_pdf_files(verbose=verbose)
            if not documents:
                return False, "No documents found or failed to load documents"
            
            # Step 2: Create chunks
            text_chunks = self.create_chunks(documents, verbose=verbose)
            if not text_chunks:
                return False, "Failed to create text chunks"
            
            # Step 3: Create vectorstore
            vectorstore = self.create_vectorstore(text_chunks, verbose=verbose)
            if vectorstore is None:
                return False, "Failed to create vectorstore"
            
            # Step 4: Save vectorstore
            if not self.save_vectorstore(vectorstore, verbose=verbose):
                return False, "Failed to save vectorstore"
            
            if verbose:
                total_time = time.time() - start_time
                print("="*50)
                print("Document Processing Complete!")
                print(f"Total time: {total_time:.2f} seconds")
                print(f"Documents processed: {len(documents)}")
                print(f"Text chunks created: {len(text_chunks)}")
                print(f"Vectorstore saved to: {self.db_path}")
                print("="*50)
            
            return True, f"Successfully processed {len(documents)} documents into {len(text_chunks)} chunks"
            
        except Exception as e:
            error_msg = f"Error in document processing pipeline: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def get_stats(self) -> dict:
        """Get statistics about the current setup"""
        stats = {
            "data_path": str(self.data_path),
            "db_path": str(self.db_path),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model_name,
            "pdf_files_count": len(list(self.data_path.glob("*.pdf"))),
            "vectorstore_exists": self.db_path.exists()
        }
        return stats

# Example usage and backward compatibility
def main():
    """Main function for standalone execution"""
    # Initialize processor
    processor = DocumentProcessor()
    
    # Print current stats
    print("Current Configuration:")
    stats = processor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Process documents
    success, message = processor.process_documents(verbose=True)
    
    if success:
        print(f"\n✅ SUCCESS: {message}")
    else:
        print(f"\n❌ ERROR: {message}")

# Backward compatibility - maintain original function names
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data_path: str = DATA_PATH) -> List[Document]:
    """Backward compatibility function"""
    processor = DocumentProcessor(data_path=data_path)
    return processor.load_pdf_files()

def create_chunks(extracted_data: List[Document]) -> List[Document]:
    """Backward compatibility function"""
    processor = DocumentProcessor()
    return processor.create_chunks(extracted_data)

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Backward compatibility function"""
    processor = DocumentProcessor()
    return processor.get_embedding_model()

# Original code execution for backward compatibility
if __name__ == "__main__":
    # Option 1: Use the new class-based approach
    print("Using Enhanced Document Processor:")
    main()
    
    print("\n" + "="*50)
    print("Using Original Approach (Backward Compatibility):")
    print("="*50)
    
    # Option 2: Use original function-based approach
    documents = load_pdf_files()


    print(f"Length of PDF pages: {len(documents)}")
    
    text_chunks = create_chunks(extracted_data=documents)
    print(f"Length of Text Chunks: {len(text_chunks)}")
    
    embedding_model = get_embedding_model()
    
    if text_chunks:
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        print(f"Vectorstore saved to: {DB_FAISS_PATH}")