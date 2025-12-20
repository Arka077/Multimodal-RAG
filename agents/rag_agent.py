"""
RAG Agent using LangGraph for orchestration
"""
import re
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from config.settings import settings
from models import GeminiClient
from retrieval import HybridRetriever


class RAGState(TypedDict):
    """State for RAG workflow"""
    query: str
    query_image_path: Optional[str]
    intent: str
    entities: List[str]
    show_visuals: bool
    show_raw_data: bool
    retrieved_chunks: List[str]
    reranked_chunks: List[str]
    context: str
    answer: str
    source_metadata: List[Dict]
    chat_history: str


class RAGAgent:
    """RAG Agent with LangGraph orchestration"""
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        retriever: HybridRetriever,
        chunk_lookup: Dict[str, Dict]
    ):
        self.gemini = gemini_client
        self.retriever = retriever
        self.chunk_lookup = chunk_lookup
        
        # LangChain Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3
        )
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("process_image", self._process_image)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Add edges
        workflow.set_entry_point("analyze_query")
        
        # Conditional edge for image processing
        workflow.add_conditional_edges(
            "analyze_query",
            lambda state: "process_image" if state.get("query_image_path") else "retrieve",
            {
                "process_image": "process_image",
                "retrieve": "retrieve"
            }
        )
        
        workflow.add_edge("process_image", "retrieve")
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: RAGState) -> RAGState:
        """Analyze query to determine intent and entities"""
        query = state["query"]
        query_lower = query.lower()
        
        print("\n" + "="*80)
        print("🔍 QUERY ANALYSIS")
        print("="*80)
        print(f"📝 Query: {query}")
        
        # Detect intent
        if any(w in query_lower for w in ['compare', 'difference', 'vs']):
            intent = 'comparison'
        elif any(w in query_lower for w in ['why', 'how', 'explain', 'analyze']):
            intent = 'analytical'
        elif any(w in query_lower for w in ['list', 'types', 'examples']):
            intent = 'enumeration'
        elif any(w in query_lower for w in ['who', 'what is', 'when', 'where']):
            intent = 'factual'
        elif any(w in query_lower for w in ['table', 'data', 'show', 'display']):
            intent = 'data_display'
        else:
            intent = 'general'
        
        print(f"🎯 Intent: {intent}")
        
        # Detect visual intent
        show_visuals = any(
            trigger in query_lower
            for trigger in ['show', 'display', 'see', 'view', 'visualize', 'screenshot', 'image']
        )
        
        print(f"👁️  Show Visuals: {show_visuals}")
        
        # Detect table display intent
        show_raw_data = any(
            trigger in query_lower
            for trigger in ['show table', 'display table', 'show data', 'show all']
        )
        
        print(f"📊 Show Raw Data: {show_raw_data}")
        
        # Extract entities (simple version - could use spaCy)
        doc = self.retriever.nlp(query)
        entities = [ent.text for ent in doc.ents]
        
        print(f"🏷️  Entities Detected: {entities if entities else 'None'}")
        
        state["intent"] = intent
        state["entities"] = entities
        state["show_visuals"] = show_visuals
        state["show_raw_data"] = show_raw_data
        
        print(f"Intent: {intent}, Entities: {entities}, Show visuals: {show_visuals}")
        
        return state
    
    def _process_image(self, state: RAGState) -> RAGState:
        """Process uploaded image"""
        image_path = state["query_image_path"]
        query = state["query"]
        
        if not image_path:
            return state
        
        # Analyze image with Gemini Vision
        if query:
            prompt = f"""Analyze this image and provide:

1. ANSWER: {query}
2. DESCRIPTION: Detailed description of everything visible

Format:
ANSWER: [answer]
DESCRIPTION: [description]"""
        else:
            prompt = "Provide a detailed description of everything visible in this image."
        
        response = self.gemini.analyze_image(image_path, prompt, max_tokens=768)
        
        # Parse response
        if "ANSWER:" in response and "DESCRIPTION:" in response:
            parts = response.split("DESCRIPTION:")
            answer = parts[0].replace("ANSWER:", "").strip()
            description = parts[1].strip()
        else:
            answer = response
            description = response
        
        # Enhance query with image description for retrieval
        if query:
            state["query"] = f"{query} {description[:200]}"
        else:
            state["query"] = description
        
        print(f"Image analysis: {answer[:100]}...")
        
        return state
    
    def _retrieve(self, state: RAGState) -> RAGState:
        """Retrieve relevant chunks"""
        query = state["query"]
        intent = state["intent"]
        
        print("\n" + "="*80)
        print("📚 CHUNK RETRIEVAL")
        print("="*80)
        
        # Determine K values based on intent
        k_map = {
            'factual': (30, 5),
            'comparison': (60, 12),
            'analytical': (50, 10),
            'enumeration': (50, 10),
            'data_display': (40, 8),
            'general': (50, 15)
        }
        initial_k, final_k = k_map.get(intent, (50, 15))
        
        print(f"🔢 Retrieval Params: initial_k={initial_k}, final_k={final_k}")
        
        # Retrieve with hybrid approach
        reranked_ids = self.retriever.retrieve(
            query,
            initial_k=initial_k,
            final_k=final_k,
            confidence_threshold=0.1
        )
        
        state["reranked_chunks"] = reranked_ids
        
        print(f"\n✅ Final Retrieved Chunks: {len(reranked_ids)}")
        print("\n🏆 TOP CHUNKS SELECTED:")
        print("-" * 80)
        for idx, cid in enumerate(reranked_ids[:5], 1):
            chunk = self.chunk_lookup.get(cid)
            if chunk:
                # Use short_summary, fallback to content, with better display
                summary = chunk.get('short_summary', '').strip()
                if not summary:
                    summary = chunk.get('content', '')[:150].strip()
                if not summary:
                    summary = "[No preview available]"
                else:
                    summary = summary[:150]
                print(f"  {idx}. [{chunk.get('type', 'text').upper()}] {summary}...")
                print(f"     Source: {chunk.get('source_file', 'Unknown')}")
        print("-" * 80)
        
        return state
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer from retrieved context"""
        query = state["query"]
        reranked_ids = state["reranked_chunks"]
        show_raw_data = state.get("show_raw_data", False)
        chat_history = state.get("chat_history", "")
        
        print("\n" + "="*80)
        print("🤖 GENERATING ANSWER")
        print("="*80)
        print(f"📄 Building context from {len(reranked_ids)} chunks...")
        
        # Build context
        context_parts = []
        source_metadata = []
        
        for idx, cid in enumerate(reranked_ids, 1):
            chunk = self.chunk_lookup.get(cid)
            if not chunk:
                continue
            
            parent_chunk = self.chunk_lookup.get(chunk.get('parent_id', ''))
            doc_context = f"[Document: {parent_chunk.get('source_file', 'Unknown')}]\n" if parent_chunk else ""
            
            # For tables, include full content if raw data display
            if show_raw_data and chunk.get('type') == 'table':
                content = doc_context + f"[Type: {chunk.get('type')}]\n" + chunk.get('content', '')
            else:
                content = doc_context + (chunk.get('detailed_summary') or chunk.get('content', ''))
            
            context_parts.append(f"[SOURCE_{idx}]\n{content}")
            source_metadata.append({
                'id': idx,
                'file': chunk.get('source_file'),
                'type': chunk.get('type'),
                'chunk_id': cid,
                'display_visual': state.get('show_visuals', False),
                'is_table': chunk.get('type') == 'table'
            })
        
        context_str = "\n\n---\n\n".join(context_parts)
        
        print(f"📝 Context size: {len(context_str)} characters")
        
        # Chat history context
        chat_context = f"\n\n### Conversation History:\n{chat_history}\n" if chat_history else ""
        
        # Build prompt based on mode
        if show_raw_data:
            system_prompt = """You are a data assistant. Display complete tables and structured data.
Rules:
1. Show COMPLETE data from tables
2. Preserve formatting exactly
3. Do NOT summarize
4. Brief 1-2 sentence context before table"""
        else:
            system_prompt = """You are a highly knowledgeable assistant. Answer using ONLY provided sources.
Rules:
1. Synthesize from multiple sources into ONE precise answer
2. Include specific facts, numbers, examples
3. Do not hallucinate
4. State limitations clearly if info missing
5. Answer ONCE - no repetition"""
        
        prompt = f"""{system_prompt}

{chat_context}
### Sources:
{context_str}

### Question:
{query}

### Your Answer:"""
        
        print(f"💭 Generating response with {settings.GEMINI_MODEL}...")
        
        # Generate with LangChain
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        print(f"✅ Generated answer ({len(answer)} characters)")
        print("\n" + "="*80)
        print("💬 MODEL OUTPUT")
        print("="*80)
        print(answer[:500] + "..." if len(answer) > 500 else answer)
        print("="*80)
        
        # Remove repetitive sections
        answer = self._remove_repetitive_content(answer)
        
        # Add source references
        citations_used = []
        for i in range(1, len(reranked_ids) + 1):
            if f"[SOURCE_{i}]" in answer:
                citations_used.append(i)
        
        if citations_used:
            print(f"\n📚 Citations Used: [SOURCE_{', '.join(map(str, citations_used))}]")
            answer += "\n\n---\n### Sources Referenced:\n"
            for meta in source_metadata:
                if meta['id'] in citations_used:
                    type_indicator = " [TABLE]" if meta.get('is_table') else ""
                    answer += f"- **SOURCE_{meta['id']}**: {meta['file']}{type_indicator} (Type: {meta['type']})\n"
        else:
            print("\n⚠️  No citations found in answer")
        
        state["answer"] = answer
        state["context"] = context_str
        state["source_metadata"] = source_metadata
        
        print("\n" + "="*80 + "\n")
        
        return state
    
    def _remove_repetitive_content(self, text: str) -> str:
        """Remove repetitive sections"""
        lines = text.split('\n')
        seen_sections = set()
        result_lines = []
        skip_until_newline = False
        
        for line in lines:
            stripped = line.strip()
            
            # Detect repetitive headers
            if any(marker in stripped for marker in ['### Final Answer:', '### Note:', '### Sources:']):
                if stripped in seen_sections:
                    skip_until_newline = True
                    continue
                seen_sections.add(stripped)
                skip_until_newline = False
            
            if not skip_until_newline:
                result_lines.append(line)
        
        cleaned = '\n'.join(result_lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def answer_question(
        self,
        query: str,
        query_image_path: Optional[str] = None,
        chat_history: str = ""
    ) -> tuple[str, str, Dict[str, str], List[Dict]]:
        # Initialize state
        initial_state = RAGState(
            query=query,
            query_image_path=query_image_path,
            intent="",
            entities=[],
            show_visuals=False,
            show_raw_data=False,
            retrieved_chunks=[],
            reranked_chunks=[],
            context="",
            answer="",
            source_metadata=[],
            chat_history=chat_history
        )
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Build debug info
        debug_info = f"""**Query Analysis**
- Intent: {final_state['intent']}
- Entities: {final_state['entities']}
- Visual Display: {'Yes' if final_state['show_visuals'] else 'No'}
- Raw Data: {'Yes' if final_state['show_raw_data'] else 'No'}

**Retrieved Chunks**: {len(final_state['reranked_chunks'])}

**Top Sources:**
"""
        
        for idx, cid in enumerate(final_state['reranked_chunks'][:5], 1):
            chunk = self.chunk_lookup.get(cid)
            if chunk:
                debug_info += f"{idx}. [{chunk['type'].upper()}] {chunk['source_file']}\n"
        
        # Build citation map
        citation_map = {
            f"SOURCE_{meta['id']}": meta['chunk_id']
            for meta in final_state['source_metadata']
        }
        
        return (
            final_state['answer'],
            debug_info,
            citation_map,
            final_state['source_metadata']
        )
