"""
Hybrid retrieval with fusion and reranking
"""
import networkx as nx
import spacy
from collections import defaultdict
from typing import List, Dict, Any, Set
from sentence_transformers import CrossEncoder

from config.settings import settings
from .vector_store import VectorStore
from .bm25_search import BM25Search


class HybridRetriever:
    """Hybrid retrieval combining vector, BM25, and knowledge graph"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_search: BM25Search,
        knowledge_graph: nx.DiGraph,
        node_metadata: Dict[str, Any],
        chunk_lookup: Dict[str, Dict],
        parent_child_map: Dict[str, List[str]]
    ):
        self.vector_store = vector_store
        self.bm25_search = bm25_search
        self.kg = knowledge_graph
        self.node_metadata = node_metadata
        self.chunk_lookup = chunk_lookup
        self.parent_child_map = parent_child_map
        
        # Load models
        self.nlp = spacy.load(settings.SPACY_MODEL)
        self.cross_encoder = CrossEncoder(
            settings.CROSS_ENCODER_MODEL,
            max_length=512
        )
        if settings.DEVICE != 'meta':
            self.cross_encoder = self.cross_encoder.to(settings.DEVICE)
    
    def retrieve(
        self,
        query: str,
        initial_k: int = 50,
        final_k: int = 15,
        confidence_threshold: float = 0.1
    ) -> List[str]:
        # Extract entities and concepts
        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents]
        
        # Extract key concepts including important keywords
        concepts = []
        for tok in doc:
            # Include nouns, proper nouns, verbs, and adjectives
            if tok.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not tok.is_stop and len(tok.text) > 2:
                concepts.append(tok.lemma_)
        
        # Add important question-related expansions
        query_lower = query.lower()
        expansions = []
        
        if 'cause' in query_lower or 'reason' in query_lower:
            expansions.extend(['reason', 'factor', 'driver', 'lead', 'result'])
        if 'effect' in query_lower or 'impact' in query_lower or 'consequence' in query_lower:
            expansions.extend(['effect', 'impact', 'consequence', 'result', 'outcome'])
        if 'solution' in query_lower or 'prevent' in query_lower or 'control' in query_lower:
            expansions.extend(['solution', 'prevention', 'control', 'strategy', 'measure'])
        
        concepts.extend(expansions)
        concepts = list(set(concepts))  # Remove duplicates
        
        print(f"🔎 Entities Extracted: {entities if entities else 'None'}")
        print(f"💡 Concepts Extracted: {concepts[:10] if concepts else 'None'}")  # Show first 10
        if len(concepts) > 10:
            print(f"   ... and {len(concepts) - 10} more")
        
        # Retrieve from different sources with query expansion
        vector_results = self.vector_store.search(query, initial_k)
        
        # Expand BM25 query with concepts (limit to avoid too long queries)
        bm25_query = query
        if concepts:
            top_concepts = concepts[:5]  # Use top 5 concepts
            bm25_query = query + " " + " ".join(top_concepts)
            print(f"🔍 Expanded BM25 Query: {bm25_query}")
        bm25_results = self.bm25_search.search(bm25_query, initial_k)
        
        kg_results = self._kg_retrieval(entities + concepts, depth=2)
        
        print(f"\n📊 Retrieval Results:")
        print(f"   Vector Search: {len(vector_results)} chunks")
        print(f"   BM25 Search: {len(bm25_results)} chunks")
        print(f"   Knowledge Graph: {len(kg_results)} chunks")
        
        # Adaptive fusion based on query characteristics
        has_entities = len(entities) > 0
        has_numbers = any(c.isdigit() for c in query)
        is_question = query.strip().endswith('?')
        
        weights = {
            'vector': 0.45,
            'bm25': 0.35,
            'kg': 0.20
        }
        
        # Boost based on query type
        if has_entities:
            weights['kg'] += 0.15
        if has_numbers:
            weights['bm25'] += 0.15
        
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        print(f"\n⚖️  Fusion Weights:")
        print(f"   Vector: {weights['vector']:.2%}")
        print(f"   BM25: {weights['bm25']:.2%}")
        print(f"   Knowledge Graph: {weights['kg']:.2%}")
        
        # Advanced RRF fusion with position-based scoring
        rrf_scores = defaultdict(float)
        
        # Reciprocal Rank Fusion with adjusted k
        k = 60
        for name, results in [('vector', vector_results), ('bm25', bm25_results), ('kg', kg_results)]:
            for rank, cid in enumerate(results, 1):
                # RRF formula: weight / (k + rank)
                rrf_scores[cid] += weights[name] * (1.0 / (k + rank))
        
        # Get top candidates and expand with context
        initial_ids = [
            cid for cid, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:25]
        ]
        
        print(f"\n🔄 RRF Fusion: {len(initial_ids)} unique chunks selected")
        
        # Add context
        contextual_ids = self._get_contextual_chunks(initial_ids)
        print(f"📚 Context Expansion: {len(contextual_ids)} total chunks (added parents/siblings)")
        
        # Rerank with confidence
        reranked_ids = self._rerank_with_confidence(
            query, contextual_ids, final_k, confidence_threshold
        )
        
        return reranked_ids
    
    def _kg_retrieval(self, entities: List[str], depth: int = 1) -> List[str]:
        related_chunks = set()
        
        if not self.kg or self.kg.number_of_nodes() == 0:
            return []
        
        # Map to canonical entities
        canonical_entities = self._get_canonical_entities(entities)
        print(f"KG search on: {canonical_entities}")
        
        for entity in canonical_entities:
            if entity in self.kg:
                # Get subgraph
                subgraph_nodes = set(nx.ego_graph(self.kg, entity, radius=depth).nodes())
                
                # Find chunks
                for node, meta in self.node_metadata.items():
                    if meta.get('canonical_name') in subgraph_nodes:
                        related_chunks.update(meta.get('source_chunks', []))
        
        return list(related_chunks)
    
    def _get_canonical_entities(self, entities: List[str]) -> List[str]:
        canonical = set()
        
        alias_to_canonical = {
            node: meta['canonical_name']
            for node, meta in self.node_metadata.items()
        }
        
        for entity in entities:
            for alias, canon in alias_to_canonical.items():
                if entity.lower() in alias.lower():
                    canonical.add(canon)
        
        return list(canonical)
    
    def _get_contextual_chunks(self, chunk_ids: List[str]) -> List[str]:
        contextual_ids = set(chunk_ids)
        
        for cid in chunk_ids:
            chunk = self.chunk_lookup.get(cid)
            if chunk and 'parent_id' in chunk:
                parent_id = chunk['parent_id']
                
                # Add parent
                if parent_id in self.chunk_lookup:
                    contextual_ids.add(parent_id)
                
                # Add siblings
                contextual_ids.update(self.parent_child_map.get(parent_id, []))
        
        return list(contextual_ids)
    
    def _rerank_with_confidence(
        self,
        query: str,
        chunk_ids: List[str],
        top_k: int,
        confidence_threshold: float
    ) -> List[str]:
        # Detect modality intent
        query_lower = query.lower()
        modality_boosts = {}
        
        if any(word in query_lower for word in ['audio', 'transcription', 'speech', 'sound']):
            modality_boosts['audio'] = 0.3
            print("Intent: audio, Entities: [], Show visuals: False")
        
        if any(word in query_lower for word in ['image', 'picture', 'diagram', 'figure', 'visual', 'show', 'visualize', 'display']):
            modality_boosts['image'] = 0.3
            print("Intent: visual, Entities: [], Show visuals: True")
        
        if any(word in query_lower for word in ['table', 'data', 'tabular', 'chart', 'graph']):
            modality_boosts['table'] = 0.3
            print("Intent: data, Entities: [], Show visuals: False")
        
        # Prepare pairs for cross-encoder
        pairs = []
        valid_ids = []
        chunk_info = []
        
        for cid in list(dict.fromkeys(chunk_ids)):  # Remove duplicates
            if cid not in self.chunk_lookup:
                continue
            
            chunk = self.chunk_lookup[cid]
            
            # Skip parent chunks and very short chunks
            if chunk.get('is_parent') or len(chunk.get('content', '')) < 20:
                continue
            
            # Prepare text for scoring
            text_parts = []
            
            if chunk.get('type') in ['table', 'image']:
                # For structured data, use detailed summary
                if chunk.get('detailed_summary'):
                    text_parts.append(chunk['detailed_summary'][:400])
            else:
                # For text, use content preferentially
                if chunk.get('content'):
                    text_parts.append(chunk['content'][:400])
                elif chunk.get('detailed_summary'):
                    text_parts.append(chunk['detailed_summary'][:400])
            
            text = " ".join(text_parts).strip()
            
            if len(text) > 10:  # Minimum text length
                pairs.append([query, text])
                valid_ids.append(cid)
                chunk_info.append(chunk)
        
        if not pairs:
            print("No valid chunks to rerank")
            return []
        
        # Score with cross-encoder
        try:
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"Reranking error: {e}. Returning top chunks by order.")
            return valid_ids[:top_k]
        
        # Apply modality boosting
        boosted_scores = []
        for i, (cid, score) in enumerate(zip(valid_ids, scores)):
            chunk_type = chunk_info[i].get('type', 'text')
            boost = modality_boosts.get(chunk_type, 0.0)
            boosted_score = score + boost
            boosted_scores.append(boosted_score)
        
        # Normalize scores to 0-1 range for consistency
        max_score = max(boosted_scores) if boosted_scores else 0
        if max_score > 0:
            normalized_scores = [s / max_score for s in boosted_scores]
        else:
            normalized_scores = boosted_scores
        
        # Filter by confidence threshold
        results = sorted(
            [(cid, norm_score) for cid, norm_score in zip(valid_ids, normalized_scores) if norm_score > confidence_threshold],
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\n🎯 Cross-Encoder Reranking:")
        print(f"   Scored {len(valid_ids)} chunks")
        print(f"   {len(results)} chunks passed threshold ({confidence_threshold:.2f})")
        
        if results:
            top_3_scores = [norm_score for _, norm_score in results[:3]]
            print(f"   Top 3 scores: {', '.join([f'{s:.3f}' for s in top_3_scores])}")
        
        # Ensure minimum chunks returned
        if len(results) < top_k // 2:
            print(f"⚠️  Only {len(results)} passed threshold, relaxing to get {top_k} results")
            results = sorted(
                zip(valid_ids, normalized_scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
        
        final_chunks = [cid for cid, _ in results[:top_k]]
        print(f"✅ Final {len(final_chunks)} chunks selected for generation")
        
        return final_chunks
