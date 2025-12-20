"""
Chat session management
"""
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from typing import Dict, List, Any, Optional

from config.settings import settings


class ChatSessionManager:
    """Manage chat sessions and history"""
    
    def __init__(self):
        self.current_session_id = None
        self.sessions = {}
    
    def create_new_session(self) -> str:
        session_id = str(uuid4())
        self.current_session_id = session_id
        
        self.sessions[session_id] = {
            'id': session_id,
            'created_at': datetime.now().isoformat(),
            'messages': [],
            'title': 'New Chat'
        }
        
        return session_id
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        if session_id not in self.sessions:
            return False
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.sessions[session_id]['messages'].append(message)
        
        # Set title from first user message
        if len(self.sessions[session_id]['messages']) == 2:
            self.sessions[session_id]['title'] = content[:50] + (
                '...' if len(content) > 50 else ''
            )
        
        self.save_session(session_id)
        return True
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, {}).get('messages', [])
    
    def get_recent_context(self, session_id: str, n_turns: int = 3) -> str:
        messages = self.get_session_history(session_id)
        recent = messages[-(n_turns * 2):]
        
        return "\n".join([
            f"{msg['role'].title()}: {msg['content'][:300]}"
            for msg in recent
        ])
    
    def save_session(self, session_id: str):
        if session_id in self.sessions:
            session_file = settings.CHAT_HISTORY_DIR / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.sessions[session_id], f, indent=2)
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        session_file = settings.CHAT_HISTORY_DIR / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file, 'r', encoding='utf-8') as f:
            self.sessions[session_id] = json.load(f)
        
        self.current_session_id = session_id
        return self.sessions[session_id]
    
    def list_all_sessions(self) -> List[Dict]:
        session_files = sorted(
            settings.CHAT_HISTORY_DIR.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        sessions_list = []
        for sf in session_files:
            try:
                with open(sf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions_list.append({
                        'id': data['id'],
                        'title': data.get('title', 'Untitled'),
                        'created_at': data.get('created_at', ''),
                        'message_count': len(data.get('messages', []))
                    })
            except:
                continue
        
        return sessions_list
