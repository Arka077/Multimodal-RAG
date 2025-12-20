"""
Text chunking strategies for document processing
Supports paragraph-based and semantic chunking with overlap
"""
import re
from typing import List


def clean_model_output(text: str) -> str:
    """Clean LLM output by removing role prefixes and formatting artifacts"""
    if not text:
        return text
    text = re.sub(
        r'^(system|user|assistant|### (Response|Instruction|Input|Your Answer)[^:]:\s)',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    return text.strip()


def semantic_chunking(text: str, max_chunk_size: int = 600, overlap: int = 100) -> List[str]:
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in sentences:
        sent_length = len(sent.split())
        
        # If adding this sentence exceeds max size and we have content, finalize chunk
        if current_length + sent_length > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Create overlap by including last N words
            overlap_sents = []
            overlap_length = 0
            for s in reversed(current_chunk):
                s_length = len(s.split())
                if overlap_length + s_length <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_length += s_length
                else:
                    break
            
            current_chunk = overlap_sents + [sent]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sent)
            current_length += sent_length
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def paragraph_chunking(
    text: str,
    max_chunk_size: int = 800,
    overlap: int = 100
) -> List[str]:
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\n+', text)
    
    # Also split by markdown headers
    all_chunks = []
    for para in paragraphs:
        if para.startswith('#'):
            # This is a header, keep it
            all_chunks.append(para)
        else:
            all_chunks.append(para)
    
    # Now group paragraphs into chunks respecting max size
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in all_chunks:
        if not para.strip():
            continue
        
        para_length = len(para.split())
        
        # If adding this paragraph exceeds max size and we have content, finalize chunk
        if current_length + para_length > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
            # Create overlap with last paragraph(s)
            overlap_paras = []
            overlap_length = 0
            for p in reversed(current_chunk):
                p_length = len(p.split())
                if overlap_length + p_length <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_length += p_length
                else:
                    break
            
            current_chunk = overlap_paras + [para]
            current_length = sum(len(p.split()) for p in current_chunk)
        else:
            current_chunk.append(para)
            current_length += para_length
    
    # Add final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks
