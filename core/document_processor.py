"""
Document processing for PDF, images, and audio files
"""
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4
from collections import defaultdict
from PIL import Image
from pytesseract import image_to_string
from pdf2image import convert_from_path
from docling.document_converter import DocumentConverter
from docling_core.types.doc import PictureItem
from tqdm.auto import tqdm

from config.settings import settings
from models import GeminiClient, WhisperClient
from .chunking import semantic_chunking, clean_model_output


class DocumentProcessor:
    """Process various document types and extract structured information"""
    
    def __init__(self, gemini_client: GeminiClient, whisper_client: WhisperClient):
        self.gemini = gemini_client
        self.whisper = whisper_client
    
    def process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        all_chunks = []
        
        print(f"\n📄 Processing {len(file_paths)} files...")
        for idx, file_path in enumerate(file_paths, 1):
            file_path = Path(file_path)
            
            print(f"\n[{idx}/{len(file_paths)}] Processing: {file_path.name}")
            
            if file_path.suffix.lower() == ".pdf":
                chunks = self._process_pdf(file_path)
            elif file_path.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
                chunks = self._process_audio(file_path)
            elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                chunks = self._process_image(file_path)
            else:
                print(f"   ❌ Unsupported file type: {file_path.suffix}")
                continue
            
            if chunks:
                print(f"   ✅ Created {len(chunks)} chunks")
                all_chunks.extend(chunks)
            else:
                print(f"   ⚠️  No chunks created from this file")
        
        print(f"\n✅ Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        elements = []
    
        try:
            print(f"   ⏳ Converting PDF (this may take 5-10 minutes for large files)...")
            
            # Configure docling to disable OCR to avoid RapidOCR permission issues
            from docling.datamodel. pipeline_options import PdfPipelineOptions
            from docling. document_converter import DocumentConverter, PdfFormatOption
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options. do_ocr = False  # Disable OCR entirely
            
            doc_converter = DocumentConverter(
                format_options={
                    "pdf":  PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            result = doc_converter.convert(str(pdf_path))
            print(f"   ✅ PDF conversion complete")
            
            full_text = result.document. export_to_markdown()
            
            # Track chunks by page for screenshot generation
            chunks_by_page = defaultdict(list)
            
            if full_text and full_text.strip():
                print(f"   📝 Extracting text chunks...")
                # Create parent chunk
                parent_id = str(uuid4())
                elements.append({
                    "type": "text",
                    "content": full_text,
                    "is_parent": True,
                    "parent_id": parent_id,
                    "chunk_id": parent_id,
                    "source_file": pdf_path.name
                })
                
                # Create child chunks
                for i, child_content in enumerate(semantic_chunking(full_text)):
                    chunk_id = str(uuid4())
                    estimated_page = i // 3  # Rough estimate
                    
                    chunk_dict = {
                        "type": "text",
                        "content": child_content,
                        "is_parent": False,
                        "parent_id": parent_id,
                        "child_index": i,
                        "chunk_id": chunk_id,
                        "page_number": estimated_page,
                        "source_file": pdf_path.name
                    }
                    elements.append(chunk_dict)
                    chunks_by_page[estimated_page].append(chunk_dict)
                
                print(f"   📄 Created {len(elements)-1} text chunks")
            
            # Generate screenshots
            print(f"   🖼️  Generating page screenshots...")
            screenshot_map = self._extract_page_screenshots(pdf_path, chunks_by_page)
            
            # Add screenshot paths
            for elem in elements:
                if elem.get('chunk_id') in screenshot_map:
                    elem['screenshot_path'] = screenshot_map[elem['chunk_id']]
            
            # Extract tables
            tables = getattr(result.document, "tables", [])
            if tables:
                print(f"   📊 Extracting {len(tables)} tables...")
                for table in tables:
                    table_csv = table.export_to_dataframe().to_csv(index=False)
                    elements.append({
                        "type": "table",
                        "content": table_csv,
                        "source_file": pdf_path.name,
                        "chunk_id": str(uuid4())
                    })
            
            # Extract images
            images = []
            for i, (item, _) in enumerate(result.document.iterate_items()):
                if isinstance(item, PictureItem):
                    try:
                        img_path = settings.IMAGES_OUTPUT_DIR / f"{pdf_path.stem}-figure-{i+1}.png"
                        item.get_image(result.document).save(img_path)
                        elements.append({
                            "type": "image",
                            "content": str(img_path),
                            "source_file": pdf_path.name,
                            "chunk_id": str(uuid4())
                        })
                        images.append(str(img_path))
                    except Exception as e:
                        print(f"   ⚠️  Error saving image: {e}")
            
            if images:
                print(f"   🖼️  Extracted {len(images)} images")
            
        except Exception as e:
            print(f"   ❌ Error processing PDF {pdf_path}: {e}")
        
        # Process chunks with Gemini
        print(f"   💬 Adding summaries...")
        return self._enrich_chunks_with_summaries(elements)
    
    def _process_audio(self, audio_path: Path) -> List[Dict[str, Any]]:
        print(f"   🎵 Transcribing audio with Whisper...")
        transcription, wav_path = self.whisper.transcribe_audio(str(audio_path))
        
        if transcription == "Transcription failed":
            print(f"   ❌ Audio transcription failed")
            return []
        
        print(f"   ✅ Transcription complete")
        
        elements = []
        parent_id = str(uuid4())
        
        # Parent chunk
        elements.append({
            "type": "audio",
            "content": transcription,
            "is_parent": True,
            "parent_id": parent_id,
            "chunk_id": parent_id,
            "audio_file": wav_path,
            "source_file": audio_path.name
        })
        
        # Child chunks
        print(f"   📝 Creating audio chunks...")
        for i, child_content in enumerate(semantic_chunking(transcription)):
            elements.append({
                "type": "audio",
                "content": child_content,
                "is_parent": False,
                "parent_id": parent_id,
                "child_index": i,
                "audio_file": wav_path,
                "source_file": audio_path.name,
                "chunk_id": str(uuid4())
            })
        
        print(f"   📄 Created {len(elements)-1} audio chunks")
        print(f"   💬 Adding summaries...")
        return self._enrich_chunks_with_summaries(elements)
    
    def _process_image(self, image_path: Path) -> List[Dict[str, Any]]:
        print(f"   🖼️  Analyzing image with Gemini Vision...")
        results = self.gemini.analyze_image_multipart(
            str(image_path),
            ["summary", "detailed_description", "ocr"]
        )
        
        print(f"   ✅ Image analysis complete")
        
        return [{
            "type": "image",
            "content": str(image_path),
            "short_summary": results.get("summary", ""),
            "detailed_summary": results.get("detailed_description", ""),
            "ocr_text": results.get("ocr", ""),
            "source_file": image_path.name,
            "chunk_id": str(uuid4()),
            "is_parent": False
        }]
    
    def _enrich_chunks_with_summaries(self, chunks: List[Dict]) -> List[Dict]:
        enriched = []
        
        print(f"   🧠 Generating detailed summaries with Gemini...")
        
        for idx, chunk in enumerate(chunks, 1):
            if chunk.get('is_parent'):
                chunk['short_summary'] = "Parent document chunk"
                chunk['detailed_summary'] = "Full parent document"
                enriched.append(chunk)
                continue
            
            # Skip if already has summaries (like images)
            if 'short_summary' in chunk and 'detailed_summary' in chunk:
                enriched.append(chunk)
                continue
            
            content = chunk['content'][:3000]  # Increased for better context
            chunk_type = chunk.get('type', 'text')
            
            # Generate comprehensive summary with Gemini for ALL chunk types
            print(f"      [{idx}/{len(chunks)}] Summarizing {chunk_type} chunk...")
            
            summary_prompt = f"""Analyze this {chunk_type} content and provide:

1. A concise 1-sentence summary (max 20 words)
2. A detailed summary covering:
   - Main topics and concepts
   - Key facts, numbers, or data points
   - Important entities (people, places, organizations, dates)
   - Relationships and connections mentioned
   - Any conclusions or implications

Content:
{content}

Provide in format:
SHORT: <1-sentence summary>
DETAILED: <comprehensive summary>"""
            
            try:
                response = self.gemini.generate_text(summary_prompt, max_tokens=500)
                
                # Parse response
                short_summary = ""
                detailed_summary = ""
                
                if "SHORT:" in response and "DETAILED:" in response:
                    parts = response.split("DETAILED:")
                    short_summary = parts[0].replace("SHORT:", "").strip()
                    detailed_summary = parts[1].strip()
                else:
                    # Fallback: use first line as short, rest as detailed
                    lines = response.strip().split('\n', 1)
                    short_summary = lines[0][:200]
                    detailed_summary = lines[1] if len(lines) > 1 else response
                
                chunk['short_summary'] = short_summary
                chunk['detailed_summary'] = detailed_summary
                chunk['gemini_enriched'] = True
                
            except Exception as e:
                print(f"        ⚠️  Error generating summary: {e}")
                # Fallback to extractive
                chunk['short_summary'] = content[:200]
                chunk['detailed_summary'] = content[:800]
                chunk['gemini_enriched'] = False
            
            enriched.append(chunk)
        
        print(f"   ✅ Completed summaries for {len(enriched)} chunks")
        return enriched
    
    def _extract_short_summary(self, text: str) -> str:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        sentences = nltk.sent_tokenize(text)
        if sentences:
            # Return first sentence, max 150 chars
            summary = sentences[0][:150]
            return summary if summary.endswith('.') else summary + '...'
        return text[:150] + '...'
    
    def _extract_detailed_summary(self, text: str) -> str:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        sentences = nltk.sent_tokenize(text)
        if len(sentences) >= 3:
            return ' '.join(sentences[:3])
        return text[:500]
    
    def _extract_page_screenshots(
        self,
        pdf_path: Path,
        chunks_by_page: Dict[int, List[Dict]]
    ) -> Dict[str, str]:
        screenshot_map = {}
        
        try:
            all_pages = convert_from_path(str(pdf_path), dpi=150)
            
            for page_num, chunks in chunks_by_page.items():
                if page_num >= len(all_pages):
                    continue
                
                page_img = all_pages[page_num]
                
                for chunk in chunks:
                    chunk_id = chunk['chunk_id']
                    screenshot_path = settings.SCREENSHOT_CACHE_DIR / f"{chunk_id}.jpg"
                    
                    if not screenshot_path.exists():
                        page_img.save(screenshot_path, "JPEG", quality=70, optimize=True)
                    
                    screenshot_map[chunk_id] = str(screenshot_path)
        
        except Exception as e:
            print(f"Error generating screenshots: {e}")
        
        return screenshot_map
