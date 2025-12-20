"""
Streamlit web interface for the RAG system
"""
import os
import sys

# Configure writable cache directories for Streamlit Cloud
os.environ['RAPIDOCR_HOME'] = '/tmp/rapidocr'
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['XDG_CACHE_HOME'] = '/tmp/cache'

# Create directories if they don't exist
for dir_path in ['/tmp/rapidocr', '/tmp/huggingface', '/tmp/torch', '/tmp/cache']:
    os. makedirs(dir_path, exist_ok=True)
import re
import streamlit as st
from typing import Dict, List
from pathlib import Path
import base64

from agents import RAGAgent
from retrieval import HybridRetriever
from utils import ChatSessionManager, GraphVisualizer, EmbeddingVisualizer
from .knowledge_base import KnowledgeBaseManager


# Page config
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-badge {
        background: #3b82f6;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        margin: 0 3px;
        font-size: 0.85em;
        font-weight: 600;
        display: inline-block;
    }
    .visual-container {
        margin: 15px 0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .image-visual {
        border: 2px solid #3b82f6;
        background: #f8fafc;
    }
    .text-visual {
        border: 2px solid #10b981;
        background: #f0fdf4;
    }
    .table-visual {
        border: 2px solid #8b5cf6;
        background: #faf5ff;
    }
    .audio-visual {
        border: 2px solid #f59e0b;
        background: #fffbeb;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 0.9em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    thead tr {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    th {
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
    }
    td {
        padding: 10px 15px;
        text-align: left;
    }
    tbody tr:nth-child(even) {
        background: #f8fafc;
    }
    tbody tr:nth-child(odd) {
        background: #ffffff;
    }
    tbody tr {
        border-bottom: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


def get_image_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""


def enhance_answer_with_visuals(answer: str, source_metadata: List[Dict]) -> str:
    if not source_metadata:
        return answer
    
    kb_manager = st.session_state.kb_manager
    
    for meta in source_metadata:
        if not meta.get('display_visual', False):
            continue
        
        chunk = kb_manager.chunk_lookup.get(meta['chunk_id'])
        if not chunk:
            continue
        
        source_ref = f"SOURCE_{meta['id']}"
        visual_html = ""
        
        # Image chunks
        if chunk['type'] == 'image' and chunk.get('content'):
            img_path = chunk['content']
            if os.path.exists(img_path):
                st.image(img_path, caption=f"📷 {chunk.get('source_file', 'Unknown')}", use_container_width=True)
        
        # Text/table chunks with screenshots
        elif chunk['type'] in ['text', 'table'] and chunk.get('screenshot_path'):
            screenshot_path = chunk['screenshot_path']
            if os.path.exists(screenshot_path):
                page_info = f" - Page {chunk.get('page_number', 'N/A')}" if 'page_number' in chunk else ""
                icon = "📄" if chunk['type'] == 'text' else "📊"
                st.image(screenshot_path, caption=f"{icon} {chunk.get('source_file', 'Unknown')}{page_info}", use_container_width=True)
        
        # Audio chunks
        elif chunk['type'] == 'audio' and chunk.get('audio_file'):
            audio_path = chunk['audio_file']
            if os.path.exists(audio_path):
                st.audio(audio_path, format='audio/wav')
                st.caption(f"🎵 {chunk.get('source_file', 'Unknown')}")
    
    return answer


def format_table_in_answer(answer: str) -> str:
    return answer


def format_citations(content: str, citation_map: dict = None) -> str:
    source_matches = re.findall(r'\[SOURCE_(\d+)\]', content)
    for match in source_matches:
        source_num = int(match)
        source_ref = f'[SOURCE_{source_num}]'
        
        # Get chunk ID from citation map if available
        chunk_id = ""
        if citation_map and f'SOURCE_{source_num}' in citation_map:
            chunk_id = citation_map[f'SOURCE_{source_num}']
            badge_html = f'<span class="source-badge" title="Chunk ID: {chunk_id}">SOURCE_{source_num}</span>'
        else:
            badge_html = f'<span class="source-badge">SOURCE_{source_num}</span>'
        
        content = content.replace(source_ref, badge_html)
    return content


def start_new_chat():
    st.session_state.messages = []
    st.session_state.current_session_id = st.session_state.chat_manager.create_new_session()
    st.session_state.citation_map = {}
    st.rerun()


def process_chat_message(message: str, query_image=None):
    if not message and not query_image:
        return
    
    display_message = message if message else "🖼️ [Image Query]"
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": display_message})
    st.session_state.chat_manager.add_message(st.session_state.current_session_id, 'user', display_message)
    
    # Get chat context
    chat_context = st.session_state.chat_manager.get_recent_context(st.session_state.current_session_id, n_turns=3)
    
    # Ask RAG agent
    try:
        with st.spinner("🔍 Searching knowledge base..."):
            answer, debug_info, citation_map, source_metadata = st.session_state.rag_agent.answer_question(
                message,
                query_image_path=query_image,
                chat_history=chat_context
            )
        
        # Store debug info and citations
        st.session_state.citation_map = citation_map
        st.session_state.last_debug_info = debug_info
        st.session_state.last_source_metadata = source_metadata
        
        # Format answer with citations (including chunk IDs)
        answer = format_citations(answer, citation_map)
        
    except Exception as e:
        answer = f"❌ Error: {str(e)}"
        import traceback
        st.error(traceback.format_exc())
    
    # Save response
    st.session_state.chat_manager.add_message(
        st.session_state.current_session_id, 
        'assistant', 
        answer, 
        {'citations': citation_map}
    )
    
    # Add assistant message to chat
    st.session_state.messages.append({"role": "assistant", "content": answer})


def get_indexed_files():
    kb_manager = st.session_state.kb_manager
    
    if not kb_manager.chunks:
        return "📭 **No files indexed yet.**\n\nUpload files to build the knowledge base."
    
    # Group chunks by source file
    files_dict = {}
    for chunk in kb_manager.chunks:
        source_file = chunk.get('source_file', 'Unknown')
        chunk_type = chunk.get('type', 'unknown').upper()
        
        if source_file not in files_dict:
            files_dict[source_file] = {'count': 0, 'types': set()}
        
        files_dict[source_file]['count'] += 1
        files_dict[source_file]['types'].add(chunk_type)
    
    # Build display
    output = f"### 📂 Currently Indexed Files\n\n**Total Files:** {len(files_dict)} | **Total Chunks:** {len(kb_manager.chunks)}\n\n"
    
    for filename, info in sorted(files_dict.items()):
        type_icons = {
            'TEXT': '📄',
            'IMAGE': '🖼️',
            'TABLE': '📊',
            'AUDIO': '🎵'
        }
        
        type_str = ', '.join([f"{type_icons.get(t, '📌')} {t}" for t in sorted(info['types'])])
        output += f"- **{filename}**\n  - Chunks: {info['count']} | Types: {type_str}\n"
    
    # Add statistics
    output += f"\n### 📊 Knowledge Base Statistics\n\n"
    output += f"- **Vector Index Size:** {kb_manager.vector_store.index.ntotal if kb_manager.vector_store.index else 0}\n"
    output += f"- **Knowledge Graph Nodes:** {kb_manager.knowledge_graph.number_of_nodes()}\n"
    output += f"- **Knowledge Graph Edges:** {kb_manager.knowledge_graph.number_of_edges()}\n"
    
    return output


def add_files(files):
    if not files:
        return "No files provided."
    
    try:
        kb_manager = st.session_state.kb_manager
        
        # Save uploaded files temporarily
        temp_paths = []
        for uploaded_file in files:
            from config.settings import settings
            temp_path = Path(settings.INPUT_DIR) / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_paths.append(str(temp_path))
        
        # Show batch processing message
        st.info(f"📚 Processing {len(temp_paths)} files...")
        
        # Process files with progress
        result = kb_manager.add_new_files_and_rebuild(temp_paths)
        
        # Update retriever with new data
        st.session_state.retriever.knowledge_graph = kb_manager.knowledge_graph
        st.session_state.retriever.node_metadata = kb_manager.node_metadata
        st.session_state.retriever.chunk_lookup = kb_manager.chunk_lookup
        st.session_state.retriever.parent_child_map = kb_manager.parent_child_map
        
        # Show completion stats
        st.success(f"✅ {result}")
        st.info(f"""
        **Processing Complete:**
        - Total chunks: {len(kb_manager.chunks)}
        - Vectors: {kb_manager.vector_store.index.ntotal if kb_manager.vector_store.index else 0}
        - KG nodes: {kb_manager.knowledge_graph.number_of_nodes()}
        - KG edges: {kb_manager.knowledge_graph.number_of_edges()}
        """)
        
        return get_indexed_files()
    except Exception as e:
        error_msg = f"Error adding files: {e}"
        st.error(error_msg)
        import traceback
        st.error(traceback.format_exc())
        return error_msg


# Main App
def main():
    # Initialize ALL session state variables IN THE CORRECT ORDER
    if 'kb_manager' not in st.session_state:
        st.session_state.kb_manager = KnowledgeBaseManager(auto_clear=True)
    
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatSessionManager()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_session_id' not in st.session_state:
        # NOW chat_manager exists, so we can use it
        st.session_state.current_session_id = st.session_state.chat_manager.create_new_session()
    
    if 'citation_map' not in st.session_state:
        st.session_state.citation_map = {}
    
    if 'retriever' not in st.session_state:
        kb = st.session_state.kb_manager
        st.session_state.retriever = HybridRetriever(
            vector_store=kb.vector_store,
            bm25_search=kb.bm25_search,
            knowledge_graph=kb.knowledge_graph,
            node_metadata=kb.node_metadata,
            chunk_lookup=kb.chunk_lookup,
            parent_child_map=kb.parent_child_map
        )
    
    if 'rag_agent' not in st.session_state:
        kb = st.session_state.kb_manager
        st.session_state.rag_agent = RAGAgent(
            gemini_client=kb.gemini,
            retriever=st.session_state.retriever,
            chunk_lookup=kb.chunk_lookup
        )
    
    # NOW start your UI code
    st.title("💬 Enhanced Multimodal RAG System")
    st.caption("*Powered by Gemini API, LangChain & LangGraph*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Navigation")
        
        tab_selection = st.radio(
            "Select Tab",
            ["💬 Chat", "📚 Knowledge Base", "💾 Chat History", "🔍 Citation Explorer", "🕸 Knowledge Graph", "📊 Embeddings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick actions
        if st.button("➕ New Chat", use_container_width=True):
            start_new_chat()
        
        st.divider()
        
        # Stats
        kb = st.session_state.kb_manager
        st.metric("📦 Total Chunks", len(kb.chunks))
        st.metric("🔢 Vector Index", kb.vector_store.index.ntotal if kb.vector_store.index else 0)
        st.metric("🕸️ KG Nodes", kb.knowledge_graph.number_of_nodes())
    
    # Main content based on selected tab
    if tab_selection == "💬 Chat": 
        render_chat_tab()
    elif tab_selection == "📚 Knowledge Base": 
        render_knowledge_base_tab()
    elif tab_selection == "💾 Chat History":
        render_chat_history_tab()
    elif tab_selection == "🔍 Citation Explorer":
        render_citation_explorer_tab()
    elif tab_selection == "🕸 Knowledge Graph":
        render_knowledge_graph_tab()
    elif tab_selection == "📊 Embeddings":
        render_embeddings_tab()

def render_chat_tab():
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
                
                # Show visuals for assistant messages
                if message["role"] == "assistant" and hasattr(st.session_state, 'last_source_metadata'):
                    enhance_answer_with_visuals(message["content"], st.session_state.last_source_metadata)
        
        # Tips
        with st.expander("💡 Tips"):
            st.markdown("""
            - For tables: "show table", "display data"
            - For images: "show image", "visualize"
            - For analysis: "why", "how", "explain"
            """)
        
        # Image upload
        query_image = st.file_uploader("📷 Upload Image to Query", type=['png', 'jpg', 'jpeg'])
        query_image_path = None
        
        if query_image:
            from config.settings import settings
            temp_path = Path(settings.INPUT_DIR) / query_image.name
            with open(temp_path, "wb") as f:
                f.write(query_image.getbuffer())
            query_image_path = str(temp_path)
            st.image(query_image, caption="Query Image", use_container_width=True)
        
        # Chat input
        if prompt := st.chat_input("Type your question..."):
            process_chat_message(prompt, query_image_path)
            st.rerun()
    
    with col2:
        st.subheader("🔍 Retrieval Details")
        if hasattr(st.session_state, 'last_debug_info'):
            st.markdown(st.session_state.last_debug_info)
        else:
            st.info("No retrieval information yet. Ask a question to see details.")


def render_knowledge_base_tab():
    
    st.subheader("📚 Knowledge Base Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📥 Upload Documents
        **Supported formats:**
        - 📄 PDF
        - 🖼️ Images (PNG, JPG)
        - 🎵 Audio (WAV, MP3, M4A, FLAC)
        
        **Processing:**
        - Text/Audio: Extractive summaries
        - Images/Tables: Gemini analysis
        - All: Paragraph-based chunking
        """)
        
        uploaded_files = st.file_uploader(
            "Upload Files",
            type=['pdf', 'png', 'jpg', 'jpeg', 'wav', 'mp3', 'm4a', 'flac'],
            accept_multiple_files=True
        )
        
        if st.button("📥 Add Files & Process", type="primary"):
            if uploaded_files:
                with st.spinner("Processing files..."):
                    result = add_files(uploaded_files)
                    st.markdown(result)
            else:
                st.warning("Please upload files first.")
    
    with col2:
        st.markdown("### 📂 Indexed Files")
        if st.button("🔄 Refresh"):
            st.rerun()
        
        st.markdown(get_indexed_files())


def render_chat_history_tab():
    
    st.subheader("💾 Chat History")
    
    if st.button("🔄 Refresh Sessions"):
        st.rerun()
    
    sessions = st.session_state.chat_manager.list_all_sessions()
    
    if not sessions:
        st.info("No saved chat sessions")
    else:
        st.markdown("### Saved Chat Sessions")
        for session in sessions[:10]:
            with st.expander(f"**{session['title']}** - {session['message_count']} messages"):
                st.text(f"ID: {session['id']}")
                if st.button(f"Load Session", key=f"load_{session['id']}"):
                    loaded_session = st.session_state.chat_manager.load_session(session['id'])
                    if loaded_session:
                        st.session_state.messages = [
                            {"role": msg['role'], "content": msg['content']}
                            for msg in loaded_session['messages']
                        ]
                        st.session_state.current_session_id = session['id']
                        st.success(f"Loaded: {session['title']}")
                        st.rerun()


def render_citation_explorer_tab():
    
    st.subheader("🔍 Citation Explorer")
    
    # Show recent citations from last query
    if hasattr(st.session_state, 'citation_map') and st.session_state.citation_map:
        st.info("💡 **Recent Citations from Last Query:**")
        for source_ref, chunk_id in st.session_state.citation_map.items():
            st.code(f"{source_ref}: {chunk_id}")
        st.divider()
    
    chunk_id = st.text_input("Enter Chunk ID", help="Copy a chunk ID from the citations above or from the chat")
    
    if st.button("🔎 Load Preview"):
        if chunk_id:
            kb_manager = st.session_state.kb_manager
            
            if chunk_id not in kb_manager.chunk_lookup:
                st.error("Chunk not found.")
            else:
                data = kb_manager.get_chunk_preview_data(chunk_id)
                
                if data:
                    st.markdown(f"## {data['source_file']}")
                    st.markdown(f"**Type:** {data['type'].upper()} | **ID:** {chunk_id[:8]}")
                    
                    if data.get('parent_summary'):
                        st.info(f"**Context:** {data['parent_summary']}")
                    
                    st.markdown("### Content")
                    
                    if data['type'] == 'image':
                        st.markdown(f"**Summary:** {data.get('short_summary','')}")
                        st.markdown(f"**Description:** {data.get('detailed_summary','')}")
                        if 'ocr_text' in data:
                            st.markdown(f"**OCR:**\n{data['ocr_text'][:500]}")
                        if os.path.exists(data['content']):
                            st.image(data['content'])
                    elif data['type'] == 'table':
                        st.markdown(data['content'][:2000])
                    else:
                        st.markdown(data['content'][:1000])
                    
                    st.json(data)
                else:
                    st.error("Unable to load chunk.")
        else:
            st.warning("Please enter a chunk ID.")


def render_knowledge_graph_tab():
    
    st.subheader("🕸 Knowledge Graph")
    st.markdown("### Visualize Knowledge Relationships")
    
    if st.button("🔄 Generate Graph"):
        with st.spinner("Generating knowledge graph..."):
            kb_manager = st.session_state.kb_manager
            visualizer = GraphVisualizer(kb_manager.knowledge_graph, kb_manager.node_metadata)
            html_content, stats = visualizer.generate_visualization()
            
            if stats:
                st.markdown(stats)
            
            if html_content:
                st.components.v1.html(html_content, height=800, scrolling=True)
            else:
                st.warning("No knowledge graph available yet. Add documents first.")


def render_embeddings_tab():
    
    st.subheader("📊 Embeddings")
    st.markdown("### 3D Embedding Space")
    
    if st.button("🔄 Generate Visualization"):
        with st.spinner("Generating embeddings visualization..."):
            kb_manager = st.session_state.kb_manager
            visualizer = EmbeddingVisualizer(
                kb_manager.vector_store.index,
                kb_manager.vector_store.chunk_ids,
                kb_manager.chunk_lookup
            )
            fig = visualizer.generate_visualization()
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No embeddings available yet. Add documents first.")


if __name__ == "__main__":
    main()
