"""
Gemini API client for text and vision tasks
Uses REST API instead of SDK for more reliable performance
"""
import requests
import json
import base64
from typing import Optional, List, Dict, Any
from pathlib import Path
from PIL import Image
import time

from config.settings import settings


class GeminiClient:
    """Unified client for Gemini API - handles both text and vision tasks with multi-key support"""
    
    def __init__(self):
        self.api_keys = settings.GEMINI_API_KEYS
        self.current_key_index = 0
        self.model_name = settings.GEMINI_MODEL
        self.base_url = "https://generativelanguage.googleapis.com/v1"
        self.embedding_model = "embedding-001"
        
        print(f"🔑 Loaded {len(self.api_keys)} API key(s)")
        
        # Generation config
        self.generation_config = {
            "temperature": settings.TEMPERATURE,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": settings.MAX_NEW_TOKENS,
        }
    
    def _get_current_api_key(self) -> str:
        return self.api_keys[self.current_key_index]
    
    def _rotate_api_key(self) -> bool:
        if len(self.api_keys) <= 1:
            print("⚠️  No additional API keys available")
            return False
        
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"🔄 Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
        return True
    
    def _is_rate_limit_error(self, error_text: str) -> bool:
        rate_limit_indicators = [
            'quota',
            'rate limit',
            'too many requests',
            '429',
            'resource_exhausted'
        ]
        error_lower = error_text.lower()
        return any(indicator in error_lower for indicator in rate_limit_indicators)
    
    def generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        retry_on_rate_limit: bool = True
    ) -> str:
        config = self.generation_config.copy()
        if max_tokens:
            config["maxOutputTokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        
        attempts = 0
        max_attempts = len(self.api_keys) if retry_on_rate_limit else 1
        
        while attempts < max_attempts:
            url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self._get_current_api_key()}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": config
            }
            
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                if "candidates" in data and len(data["candidates"]) > 0:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                return ""
                
            except requests.exceptions.RequestException as e:
                error_text = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    error_text += " " + e.response.text
                
                # Check if it's a rate limit error
                if self._is_rate_limit_error(error_text) and retry_on_rate_limit:
                    print(f"⚠️  Rate limit hit on key {self.current_key_index + 1}")
                    if self._rotate_api_key():
                        attempts += 1
                        time.sleep(1)  # Brief pause before retry
                        continue
                
                print(f"Error generating text: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text}")
                return ""
            
            attempts += 1
        
        print("❌ All API keys exhausted")
        return ""
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        retry_on_rate_limit: bool = True
    ) -> str:
        config = self.generation_config.copy()
        if max_tokens:
            config["maxOutputTokens"] = max_tokens
        
        try:
            # Read and encode image
            with open(image_path, "rb") as img_file:
                image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
            
            # Determine image media type
            suffix = Path(image_path).suffix.lower()
            media_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            media_type = media_type_map.get(suffix, "image/jpeg")
            
            url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": media_type,
                                "data": image_data
                            }
                        }
                    ]
                }],
                "generationConfig": config
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Error analyzing image {image_path}: {e}")
            return ""
        except Exception as e:
            print(f"Error analyzing image {image_path}: {e}")
            return ""
    
    def analyze_image_multipart(
        self,
        image_path: str,
        prompts: List[str]
    ) -> Dict[str, str]:
        try:
            # Read and encode image
            with open(image_path, "rb") as img_file:
                image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
            
            # Determine image media type
            suffix = Path(image_path).suffix.lower()
            media_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            media_type = media_type_map.get(suffix, "image/jpeg")
            
            # Combine prompts into structured request
            combined_prompt = """Analyze this image and provide the following:

"""
            for i, prompt_key in enumerate(prompts, 1):
                if prompt_key == "summary":
                    combined_prompt += f"{i}. SUMMARY: Provide a concise one-line summary\n"
                elif prompt_key == "detailed_description":
                    combined_prompt += f"{i}. DETAILED_DESCRIPTION: Provide a comprehensive description of all visual elements\n"
                elif prompt_key == "ocr":
                    combined_prompt += f"{i}. OCR_TEXT: Extract all readable text from the image\n"
            
            combined_prompt += "\nFormat your response clearly with headers for each section."
            
            url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": combined_prompt},
                        {
                            "inlineData": {
                                "mimeType": media_type,
                                "data": image_data
                            }
                        }
                    ]
                }],
                "generationConfig": self.generation_config
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            if "candidates" not in data or len(data["candidates"]) == 0:
                return {
                    "summary": "Analysis failed",
                    "detailed_description": "Analysis failed",
                    "ocr": ""
                }
            
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            results = {}
            
            # Parse structured response
            if "SUMMARY:" in text:
                results["summary"] = text.split("SUMMARY:")[1].split("DETAILED_DESCRIPTION:")[0].strip() if "DETAILED_DESCRIPTION:" in text else text.split("SUMMARY:")[1].strip()
            
            if "DETAILED_DESCRIPTION:" in text:
                results["detailed_description"] = text.split("DETAILED_DESCRIPTION:")[1].split("OCR_TEXT:")[0].strip() if "OCR_TEXT:" in text else text.split("DETAILED_DESCRIPTION:")[1].strip()
            
            if "OCR_TEXT:" in text:
                results["ocr"] = text.split("OCR_TEXT:")[1].strip()
            
            # Fallback if parsing fails
            if not results:
                results = {
                    "summary": text[:200],
                    "detailed_description": text,
                    "ocr": ""
                }
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error in multipart image analysis: {e}")
            return {
                "summary": "Analysis failed",
                "detailed_description": "Analysis failed",
                "ocr": ""
            }
        except Exception as e:
            print(f"Error in multipart image analysis: {e}")
            return {
                "summary": "Analysis failed",
                "detailed_description": "Analysis failed",
                "ocr": ""
            }
    
    def embed_text(self, text: str) -> List[float]:
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/models/{self.embedding_model}:embedContent?key={self.api_key}"
                
                payload = {
                    "model": self.embedding_model,
                    "content": {
                        "parts": [{
                            "text": text
                        }]
                    }
                }
                
                response = requests.post(url, json=payload)
                
                # Handle rate limiting with retry
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        print(f"Rate limited (429). Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        print(f"Rate limited (429) - max retries exceeded")
                        return []
                
                response.raise_for_status()
                
                data = response.json()
                if "embedding" in data:
                    return data["embedding"]["values"]
                return []
            except requests.exceptions.RequestException as e:
                print(f"Error generating embedding: {e}")
                return []
        
        return []
    
    def embed_texts_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                url = f"{self.base_url}/models/{self.embedding_model}:batchEmbedContents?key={self.api_key}"
                
                requests_payload = [
                    {
                        "model": self.embedding_model,
                        "content": {
                            "parts": [{
                                "text": text
                            }]
                        }
                    }
                    for text in batch
                ]
                
                payload = {
                    "requests": requests_payload
                }
                
                response = requests.post(url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                if "embeddings" in data:
                    for emb in data["embeddings"]:
                        embeddings.append(emb["values"])
                
                # Rate limiting
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"Error in batch embedding: {e}")
                # Add zero vectors for failed batches
                embeddings.extend([[0.0] * 768 for _ in batch])
        
        return embeddings
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        try:
            # Convert messages to Gemini format
            history = []
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                history.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
            
            # Add system prompt to first message if provided
            last_message_content = messages[-1]["content"]
            if system_prompt:
                last_message_content = f"{system_prompt}\n\n{last_message_content}"
            
            parts = [{"text": last_message_content}]
            
            url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": history + [{
                    "role": "user",
                    "parts": parts
                }],
                "generationConfig": self.generation_config
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            return ""
            
        except requests.exceptions.RequestException as e:
            print(f"Error in chat: {e}")
            return ""


# Global client instance
gemini_client = GeminiClient()
