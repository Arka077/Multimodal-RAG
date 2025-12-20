"""
Advanced BM25 keyword search with preprocessing and fuzzy matching
"""
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import re
from collections import Counter
import nltk
from nltk.stem import PorterStemmer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError: 
    nltk.download('punkt_tab', quiet=True)

class BM25Search:
    """BM25-based keyword search with stemming and fuzzy matching"""
    
    def __init__(self):
        self.index = None
        self.chunk_ids = []
        self.corpus = []
        self.raw_corpus = []
        self.stemmer = PorterStemmer()
    
    def _preprocess_text(self, text: str) -> List[str]:
        # Lowercase for case-insensitive matching
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^a-z0-9\s\-_]', ' ', text)
        
        # Split and filter
        tokens = text.split()
        
        # Minimal stopwords - keep important words for scientific content
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'of', 'by', 'with', 'from', 'is', 'are', 'am', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'as', 'it', 'was', 'were'
        }
        
        # Keep tokens that are:
        # - Longer than 2 characters OR
        # - Important question words (what, why, how)
        # - Not in stopwords
        tokens = [
            t for t in tokens 
            if (len(t) > 2 or t in ['why', 'how', 'who']) and t not in stopwords
        ]
        
        # Apply stemming for fuzzy matching (solution, solutions -> solut)
        # This allows ~85% match between variants
        stemmed_tokens = [self.stemmer.stem(t) for t in tokens]
        
        return stemmed_tokens
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        print("Building BM25 index...")
        
        corpus = []
        ids = []
        raw_texts = []
        
        for chunk in chunks:
            # Combine all text fields with weights
            text_parts = [
                chunk.get("detailed_summary", ""),
                chunk.get("short_summary", ""),
                chunk.get("content", "")
            ]
            
            # Add OCR text from images with lower weight
            if chunk.get("type") == "image" and "ocr_text" in chunk:
                text_parts.append(" ".join([chunk["ocr_text"]] * 1))  # Weight once
            
            text = " ".join([t for t in text_parts if t]).strip()
            
            if text:
                tokens = self._preprocess_text(text)
                if tokens:  # Only add if has tokens after preprocessing
                    corpus.append(tokens)
                    ids.append(chunk['chunk_id'])
                    raw_texts.append(text)
        
        if corpus:
            self.index = BM25Okapi(corpus, k1=2.0, b=0.75)
            self.chunk_ids = ids
            self.corpus = corpus
            self.raw_corpus = raw_texts
            print(f"BM25 index built with {len(ids)} documents")
    
    def search(self, query: str, top_k: int = 50) -> List[str]:
        if not self.index:
            return []
        
        # Preprocess query
        tokenized_query = self._preprocess_text(query)
        
        if not tokenized_query:
            return []
        
        # Get scores
        scores = self.index.get_scores(tokenized_query)
        
        # Get top indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Filter by minimum score
        min_score = 0.1
        valid_indices = [i for i in top_indices if scores[i] > min_score]
        
        return [self.chunk_ids[i] for i in valid_indices if i < len(self.chunk_ids)]
