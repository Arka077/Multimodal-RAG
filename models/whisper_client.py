"""
Whisper client for audio transcription
"""
import gc
import whisper
from pathlib import Path
from pydub import AudioSegment
from typing import Optional

from config.settings import settings


class WhisperClient:
    """Client for OpenAI Whisper audio transcription"""
    
    def __init__(self, model_size: Optional[str] = None):
        self.model_size = model_size or settings.WHISPER_MODEL_SIZE
        self.model = None
        # Don't load model at initialization - use lazy loading
    
    def _load_model(self):
        """Lazy load Whisper model only when needed"""
        if self.model is None:
            print(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
            print("Whisper model loaded successfully")
    
    def unload_model(self):
        """Unload Whisper model to free memory"""
        if self.model is not None:
            print("Unloading Whisper model...")
            del self.model
            self.model = None
            gc.collect()
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print("Whisper model unloaded")
    
    def transcribe_audio(self, audio_path: str) -> tuple[str, str]:
        try:
            # Ensure model is loaded
            self._load_model()
            
            audio_path = Path(audio_path)
            audio_filename = audio_path.stem
            
            # Convert to WAV for compatibility
            wav_path = settings.AUDIO_CACHE_DIR / f"{audio_filename}.wav"
            
            if not wav_path.exists():
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(wav_path, format="wav")
            
            # Transcribe
            result = self.model.transcribe(str(wav_path))
            transcription = result["text"]
            
            print(f"Transcribed audio: {transcription[:100]}...")
            
            return transcription, str(wav_path)
            
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {e}")
            return "Transcription failed", None
    
    def transcribe_with_timestamps(self, audio_path: str) -> dict:
        try:
            # Ensure model is loaded
            self._load_model()
            
            result = self.model.transcribe(
                str(audio_path),
                word_timestamps=True
            )
            return result
        except Exception as e:
            print(f"Error in timestamp transcription: {e}")
            return {"text": "Transcription failed", "segments": []}


# Global client instance - model will be loaded on demand
whisper_client = WhisperClient()
