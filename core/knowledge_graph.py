"""
Knowledge graph extraction and management using rule-based methods with enhanced consolidation
"""
import re
import json
import networkx as nx
import spacy
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Set
from pathlib import Path
from tqdm.auto import tqdm
from difflib import SequenceMatcher
import numpy as np

from config.settings import settings


class NodeMetadataTracker:
    """Track metadata for knowledge graph nodes"""
    
    def __init__(self):
        self.node_data = defaultdict(lambda: {
            "appearances": 0,
            "source_chunks": set(),
            "chunk_types": set(),
            "contexts": [],
            "predicates_used": set(),
            "connected_images": set(),
            "llm_type": None,
            "canonical_name": None
        })
    
    def add_node_occurrence(
        self,
        node_name: str,
        chunk_id: str,
        chunk_type: str,
        predicate: str = None,
        context: str = ""
    ):
        data = self.node_data[node_name.strip()]
        data["appearances"] += 1
        data["source_chunks"].add(chunk_id)
        data["chunk_types"].add(chunk_type)
        
        if predicate:
            data["predicates_used"].add(predicate)
        
        if chunk_type == "image":
            data["connected_images"].add(chunk_id)
        
        if context and len(data["contexts"]) < 3:
            data["contexts"].append(context[:300])
    
    def get_node_metadata(self, node_name: str) -> Dict[str, Any]:
        data = self.node_data[node_name.strip()]
        return {
            "type": data.get("llm_type", "entity"),
            "canonical_name": data.get("canonical_name", node_name.strip()),
            "frequency": data["appearances"],
            "source_chunks": list(data["source_chunks"]),
            "chunk_types": list(data["chunk_types"]),
            "has_visual_representation": len(data["connected_images"]) > 0,
            "predicates": list(data["predicates_used"]),
        }


class EdgeConsolidator:
    """Consolidate and merge similar edges in the knowledge graph"""
    
    def __init__(self, nlp):
        self.nlp = nlp
        
        # Predicate synonym mapping for normalization
        self.predicate_synonyms = {
            # Causal relationships
            'cause': ['causes', 'caused', 'causing', 'lead', 'leads', 'led', 'leading', 
                     'result', 'results', 'resulted', 'resulting', 'trigger', 'triggers', 
                     'triggered', 'produce', 'produces', 'produced', 'drive', 'drives', 
                     'driven', 'contribute', 'contributes', 'contributed'],
            
            # Effect relationships
            'affect': ['affects', 'affected', 'affecting', 'impact', 'impacts', 'impacted', 
                      'impacting', 'influence', 'influences', 'influenced', 'influencing'],
            
            # Change relationships
            'change': ['changes', 'changed', 'changing', 'alter', 'alters', 'altered', 
                      'altering', 'modify', 'modifies', 'modified', 'modifying', 'transform', 
                      'transforms', 'transformed'],
            
            # Composition relationships
            'part_of': ['include', 'includes', 'included', 'including', 'contain', 'contains', 
                       'contained', 'containing', 'consist', 'consists', 'composed', 'comprise', 
                       'comprises', 'made_of'],
            
            # Attribute relationships
            'has_attribute': ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'has', 'have', 
                            'had', 'having'],
            
            # Location relationships
            'located_in': ['located', 'locates', 'found', 'find', 'situated', 'position', 
                          'positioned', 'place', 'placed'],
            
            # Temporal relationships
            'precede': ['precedes', 'preceded', 'before', 'prior', 'earlier'],
            'follow': ['follows', 'followed', 'after', 'subsequent', 'later'],
            
            # Association relationships
            'related_to': ['relate', 'relates', 'related', 'relating', 'associate', 'associates', 
                          'associated', 'associating', 'connect', 'connects', 'connected', 
                          'connecting', 'link', 'links', 'linked', 'linking'],
            
            # Creation relationships
            'create': ['creates', 'created', 'creating', 'generate', 'generates', 'generated', 
                      'generating', 'produce', 'produces', 'produced', 'producing', 'make', 
                      'makes', 'made', 'making'],
            
            # Usage relationships
            'use': ['uses', 'used', 'using', 'employ', 'employs', 'employed', 'employing', 
                   'utilize', 'utilizes', 'utilized', 'utilizing', 'apply', 'applies', 
                   'applied', 'applying'],
            
            # Increase/Decrease relationships
            'increase': ['increases', 'increased', 'increasing', 'raise', 'raises', 'raised', 
                        'raising', 'grow', 'grows', 'grew', 'growing', 'enhance', 'enhances', 
                        'enhanced'],
            'decrease': ['decreases', 'decreased', 'decreasing', 'reduce', 'reduces', 'reduced', 
                        'reducing', 'lower', 'lowers', 'lowered', 'lowering', 'diminish', 
                        'diminishes', 'diminished'],
            
            # Ownership/Possession
            'own': ['owns', 'owned', 'owning', 'possess', 'possesses', 'possessed', 'possessing', 
                   'belong', 'belongs', 'belonged', 'belonging'],
            
            # Comparison
            'similar_to': ['similar', 'like', 'alike', 'resemble', 'resembles', 'resembled', 
                          'comparable', 'compare', 'compares'],
            'different_from': ['different', 'differ', 'differs', 'differed', 'unlike', 'contrast', 
                              'contrasts', 'contrasted'],
        }
        
        # Reverse mapping for quick lookup
        self.predicate_to_canonical = {}
        for canonical, synonyms in self.predicate_synonyms.items():
            for syn in synonyms:
                self.predicate_to_canonical[syn] = canonical
            self.predicate_to_canonical[canonical] = canonical
    
    def normalize_predicate(self, predicate: str) -> str:
        """Normalize a predicate to its canonical form"""
        predicate_lower = predicate.lower().strip()
        
        # Lemmatize the predicate
        doc = self.nlp(predicate_lower)
        if len(doc) > 0:
            lemma = doc[0].lemma_
        else:
            lemma = predicate_lower
        
        # Check if it's in our synonym mapping
        if lemma in self.predicate_to_canonical:
            return self.predicate_to_canonical[lemma]
        elif predicate_lower in self.predicate_to_canonical:
            return self.predicate_to_canonical[predicate_lower]
        
        # Return lemmatized form if not in mapping
        return lemma
    
    def consolidate_edges(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Consolidate edges by merging similar predicates between same node pairs
        """
        print("Consolidating edges...")
        
        # Dictionary to store consolidated edges: (source, target) -> {predicate: count}
        edge_groups = defaultdict(lambda: defaultdict(int))
        edge_sources = defaultdict(lambda: defaultdict(list))  # Track source chunks
        
        # Group edges by node pairs
        for source, target, data in graph.edges(data=True):
            predicate = data.get('label', 'related_to')
            normalized_pred = self.normalize_predicate(predicate)
            
            edge_groups[(source, target)][normalized_pred] += 1
            if 'source_chunks' in data:
                edge_sources[(source, target)][normalized_pred].extend(data['source_chunks'])
        
        # Build consolidated graph
        consolidated_graph = nx.DiGraph()
        
        # Add all nodes with their attributes
        for node, attrs in graph.nodes(data=True):
            consolidated_graph.add_node(node, **attrs)
        
        # Add consolidated edges
        for (source, target), predicates in edge_groups.items():
            # Choose the most frequent predicate as the canonical one
            canonical_predicate = max(predicates.items(), key=lambda x: x[1])[0]
            total_count = sum(predicates.values())
            
            # Collect all source chunks
            all_sources = []
            for pred, sources in edge_sources[(source, target)].items():
                all_sources.extend(sources)
            
            consolidated_graph.add_edge(
                source, 
                target, 
                label=canonical_predicate,
                weight=total_count,
                merged_predicates=list(predicates.keys()),
                source_chunks=list(set(all_sources))
            )
        
        original_edges = graph.number_of_edges()
        consolidated_edges = consolidated_graph.number_of_edges()
        reduction = ((original_edges - consolidated_edges) / original_edges * 100) if original_edges > 0 else 0
        
        print(f"Edge consolidation complete:")
        print(f"  Original edges: {original_edges}")
        print(f"  Consolidated edges: {consolidated_edges}")
        print(f"  Reduction: {reduction:.1f}%")
        
        return consolidated_graph
    
    def detect_redundant_paths(self, graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
        """
        Identify redundant paths where A->B->C and A->C exist with similar predicates
        Returns list of edges that could potentially be removed
        """
        print("Detecting redundant paths...")
        redundant_edges = []
        
        for node in tqdm(list(graph.nodes()), desc="Checking paths"):
            # Get all paths of length 2 from this node
            for target in graph.successors(node):
                for intermediate in graph.successors(target):
                    if intermediate == node:  # Skip cycles back to source
                        continue
                    
                    # Check if direct edge exists
                    if graph.has_edge(node, intermediate):
                        # Get predicates
                        direct_pred = graph[node][intermediate].get('label', '')
                        indirect_pred1 = graph[node][target].get('label', '')
                        indirect_pred2 = graph[target][intermediate].get('label', '')
                        
                        # Check if predicates are related (same or similar)
                        if (direct_pred == indirect_pred1 or 
                            direct_pred == indirect_pred2 or
                            indirect_pred1 == indirect_pred2):
                            redundant_edges.append((node, intermediate, direct_pred))
        
        print(f"Found {len(redundant_edges)} potentially redundant edges")
        return redundant_edges


class SemanticNodeMerger:
    """Merge semantically similar nodes using embeddings and context"""
    
    def __init__(self, nlp):
        self.nlp = nlp
    
    def compute_node_similarity(self, node1: str, node2: str, 
                                contexts1: List[str], contexts2: List[str]) -> float:
        """
        Compute semantic similarity between two nodes using:
        1. String similarity
        2. Embedding similarity
        3. Context overlap
        """
        # String similarity
        string_sim = SequenceMatcher(None, node1.lower(), node2.lower()).ratio()
        
        # Embedding similarity
        doc1 = self.nlp(node1)
        doc2 = self.nlp(node2)
        
        if doc1.has_vector and doc2.has_vector:
            embedding_sim = doc1.similarity(doc2)
        else:
            embedding_sim = 0.0
        
        # Context similarity (if available)
        context_sim = 0.0
        if contexts1 and contexts2:
            # Compare contexts using embeddings
            ctx1_text = " ".join(contexts1[:3])[:1000]
            ctx2_text = " ".join(contexts2[:3])[:1000]
            
            ctx1_doc = self.nlp(ctx1_text)
            ctx2_doc = self.nlp(ctx2_text)
            
            if ctx1_doc.has_vector and ctx2_doc.has_vector:
                context_sim = ctx1_doc.similarity(ctx2_doc)
        
        # Weighted combination
        final_sim = (string_sim * 0.3 + embedding_sim * 0.5 + context_sim * 0.2)
        
        return final_sim
    
    def detect_hierarchical_relationships(self, nodes: List[str]) -> Dict[str, str]:
        """
        Detect hierarchical relationships where specific terms should merge into general ones
        E.g., "diesel emissions" -> "emissions"
        """
        print("Detecting hierarchical relationships...")
        hierarchy_map = {}
        
        # Sort by length (longer terms might be more specific)
        sorted_nodes = sorted(nodes, key=len, reverse=True)
        
        for i, specific_node in enumerate(tqdm(sorted_nodes, desc="Finding hierarchies")):
            specific_words = set(specific_node.lower().split())
            
            # Look for potential parent terms
            for general_node in sorted_nodes[i+1:]:
                general_words = set(general_node.lower().split())
                
                # Check if general term is subset of specific term
                if general_words.issubset(specific_words) and len(general_words) > 0:
                    # Calculate how much more specific the term is
                    specificity_ratio = len(specific_words) / len(general_words)
                    
                    # Only merge if reasonably more specific (not just one extra word on very short terms)
                    if specificity_ratio <= 2.0 or len(specific_words) - len(general_words) <= 2:
                        # Check if the general term appears at start or end (more likely to be the core concept)
                        specific_lower = specific_node.lower()
                        general_lower = general_node.lower()
                        
                        if (specific_lower.endswith(general_lower) or 
                            specific_lower.startswith(general_lower) or
                            general_lower in specific_lower):
                            hierarchy_map[specific_node] = general_node
                            break
        
        print(f"Found {len(hierarchy_map)} hierarchical relationships")
        return hierarchy_map
    
    def find_semantic_clusters(self, node_tracker: NodeMetadataTracker, 
                              similarity_threshold: float = 0.85) -> Dict[str, str]:
        """
        Find clusters of semantically similar nodes and merge them
        """
        print("Finding semantic clusters...")
        
        nodes = list(node_tracker.node_data.keys())
        merge_map = {}
        processed = set()
        
        for i, node1 in enumerate(tqdm(nodes, desc="Clustering nodes")):
            if node1 in processed:
                continue
            
            data1 = node_tracker.node_data[node1]
            contexts1 = data1.get("contexts", [])
            
            cluster = [node1]
            
            for node2 in nodes[i+1:]:
                if node2 in processed:
                    continue
                
                data2 = node_tracker.node_data[node2]
                contexts2 = data2.get("contexts", [])
                
                # Compute similarity
                similarity = self.compute_node_similarity(node1, node2, contexts1, contexts2)
                
                if similarity >= similarity_threshold:
                    cluster.append(node2)
                    processed.add(node2)
            
            # If we found a cluster, choose canonical form
            if len(cluster) > 1:
                # Choose the most frequent node as canonical
                canonical = max(cluster, key=lambda n: node_tracker.node_data[n]["appearances"])
                
                for node in cluster:
                    if node != canonical:
                        merge_map[node] = canonical
            
            processed.add(node1)
        
        print(f"Found {len(merge_map)} nodes to merge in semantic clusters")
        return merge_map


class KnowledgeGraphBuilder:
    """Build knowledge graph from processed documents using rule-based methods"""
    
    def __init__(self):
        print("Loading SpaCy model for knowledge graph extraction...")
        self.nlp = spacy.load(settings.SPACY_MODEL)
        self.node_tracker = NodeMetadataTracker()
        self.graph = nx.DiGraph()
        
        # Initialize consolidation modules
        self.edge_consolidator = EdgeConsolidator(self.nlp)
        self.node_merger = SemanticNodeMerger(self.nlp)
        
        # Domain-specific patterns for relationships
        self.causal_patterns = [
            'cause', 'causes', 'caused', 'causing',
            'lead', 'leads', 'led', 'leading',
            'result', 'results', 'resulted', 'resulting',
            'trigger', 'triggers', 'triggered',
            'produce', 'produces', 'produced',
            'drive', 'drives', 'driven',
            'contribute', 'contributes', 'contributed'
        ]
        
        self.effect_patterns = [
            'affect', 'affects', 'affected',
            'impact', 'impacts', 'impacted',
            'influence', 'influences', 'influenced',
            'change', 'changes', 'changed',
            'alter', 'alters', 'altered'
        ]
        
        self.part_of_patterns = [
            'include', 'includes', 'included',
            'contain', 'contains', 'contained',
            'consist', 'consists', 'composed',
            'comprise', 'comprises', 'part of'
        ]
        
        self.attribute_patterns = [
            'is', 'are', 'was', 'were', 'be', 'been',
            'has', 'have', 'had'
        ]
    
    def build_knowledge_graph(
        self,
        chunks: List[Dict[str, Any]],
        enable_consolidation: bool = True,
        semantic_similarity_threshold: float = 0.85
    ) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        print("Building knowledge graph...")
        
        # Stage 1: Extract triples
        all_triple_contexts = self._extract_triples(chunks)
        
        # Stage 2: Classify nodes
        nodes_with_ctx = [
            (n, d["predicates_used"], d["chunk_types"], d["contexts"])
            for n, d in self.node_tracker.node_data.items()
        ]
        classifications = self._classify_nodes(nodes_with_ctx)
        
        for node, node_type in classifications.items():
            if node in self.node_tracker.node_data:
                self.node_tracker.node_data[node]["llm_type"] = node_type
        
        # Stage 3: Detect aliases (basic)
        aliases_map = self._detect_aliases(list(self.node_tracker.node_data.keys()))
        
        for original, canonical in aliases_map.items():
            if original in self.node_tracker.node_data:
                self.node_tracker.node_data[original]["canonical_name"] = canonical
        
        # Stage 4: Build initial graph
        self._build_initial_graph(all_triple_contexts)
        
        print(f"Initial graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Stage 5: Enhanced consolidation (NEW)
        if enable_consolidation:
            print("\n=== Starting Graph Consolidation ===")
            
            # 5a: Consolidate edges with similar predicates
            self.graph = self.edge_consolidator.consolidate_edges(self.graph)
            
            # 5b: Detect and optionally remove redundant paths
            redundant_edges = self.edge_consolidator.detect_redundant_paths(self.graph)
            # Note: We're just detecting, not removing. You can decide what to do with these.
            
            # 5c: Find hierarchical relationships
            hierarchy_map = self.node_merger.detect_hierarchical_relationships(
                list(self.graph.nodes())
            )
            
            # 5d: Find semantic clusters
            semantic_map = self.node_merger.find_semantic_clusters(
                self.node_tracker, 
                similarity_threshold=semantic_similarity_threshold
            )
            
            # 5e: Merge nodes based on hierarchy and semantic similarity
            all_merges = {**hierarchy_map, **semantic_map}
            if all_merges:
                self.graph = self._merge_nodes(self.graph, all_merges)
                
                # Update node tracker canonical names
                for original, canonical in all_merges.items():
                    if original in self.node_tracker.node_data:
                        self.node_tracker.node_data[original]["canonical_name"] = canonical
            
            print(f"\nFinal graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            print("=== Consolidation Complete ===\n")
        
        # Get metadata
        node_metadata = {
            n: self.node_tracker.get_node_metadata(n)
            for n in self.node_tracker.node_data
        }
        
        # Save consolidated graph
        self._save_consolidated_graph()
        
        return self.graph, node_metadata
    
    def _merge_nodes(self, graph: nx.DiGraph, merge_map: Dict[str, str]) -> nx.DiGraph:
        """
        Merge nodes in the graph according to merge_map
        """
        print(f"Merging {len(merge_map)} nodes...")
        
        merged_graph = nx.DiGraph()
        
        # Add all nodes that aren't being merged away
        for node, attrs in graph.nodes(data=True):
            if node not in merge_map:
                merged_graph.add_node(node, **attrs)
        
        # Add canonical nodes for merged nodes (if not already present)
        for canonical in set(merge_map.values()):
            if not merged_graph.has_node(canonical):
                if graph.has_node(canonical):
                    merged_graph.add_node(canonical, **graph.nodes[canonical])
                else:
                    # Create node if it doesn't exist
                    merged_graph.add_node(canonical)
        
        # Redirect edges
        for source, target, data in graph.edges(data=True):
            # Map source and target to their canonical forms
            canonical_source = merge_map.get(source, source)
            canonical_target = merge_map.get(target, target)
            
            # Skip self-loops
            if canonical_source == canonical_target:
                continue
            
            # If edge already exists, merge the weights and source chunks
            if merged_graph.has_edge(canonical_source, canonical_target):
                existing_data = merged_graph[canonical_source][canonical_target]
                existing_weight = existing_data.get('weight', 1)
                new_weight = data.get('weight', 1)
                
                existing_sources = existing_data.get('source_chunks', [])
                new_sources = data.get('source_chunks', [])
                
                merged_graph[canonical_source][canonical_target]['weight'] = existing_weight + new_weight
                merged_graph[canonical_source][canonical_target]['source_chunks'] = list(
                    set(existing_sources + new_sources)
                )
            else:
                merged_graph.add_edge(canonical_source, canonical_target, **data)
        
        return merged_graph
    
    def _extract_triples(
        self,
        chunks: List[Dict]
    ) -> List[Tuple[Tuple[str, str, str], str, str]]:
        all_triple_contexts = []
        
        for chunk in tqdm(chunks, desc="Extracting triples"):
            if chunk.get('is_parent'):
                continue
            
            combined_text = " ".join([
                chunk.get(k, "")
                for k in ["detailed_summary", "short_summary", "content"]
            ]).strip()
            
            if not combined_text or len(combined_text) < 20:
                continue
            
            chunk_type = chunk.get("type", "text")
            chunk_id = chunk['chunk_id']
            
            # Process with SpaCy
            doc = self.nlp(combined_text[:5000])  # Limit text length
            
            # Extract triples using dependency parsing
            triples = self._extract_svo_triples(doc)
            
            # Extract pattern-based triples
            triples.extend(self._extract_pattern_triples(doc))
            
            # Extract entity relationships
            triples.extend(self._extract_entity_relations(doc))
            
            # Add to contexts
            for s, p, o in triples:
                if len(s) > 2 and len(o) > 2:  # Filter very short entities
                    triple = (s, p, o)
                    all_triple_contexts.append((triple, chunk_id, chunk_type))
                    
                    self.node_tracker.add_node_occurrence(
                        s, chunk_id, chunk_type, p, combined_text
                    )
                    self.node_tracker.add_node_occurrence(
                        o, chunk_id, chunk_type, p, combined_text
                    )
        
        return all_triple_contexts
    
    def _extract_svo_triples(self, doc) -> List[Tuple[str, str, str]]:
        triples = []
        
        for token in doc:
            # Look for verbs
            if token.pos_ == "VERB":
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "attr", "oprd")]
                
                for subj in subjects:
                    for obj in objects:
                        # Get compound subjects/objects
                        subj_phrase = self._get_noun_phrase(subj)
                        obj_phrase = self._get_noun_phrase(obj)
                        predicate = token.lemma_
                        
                        if subj_phrase and obj_phrase:
                            triples.append((subj_phrase, predicate, obj_phrase))
        
        return triples
    
    def _extract_pattern_triples(self, doc) -> List[Tuple[str, str, str]]:
        triples = []
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Causal relationships
            for pattern in self.causal_patterns:
                if pattern in sent_text:
                    triple = self._extract_causal_triple(sent, pattern)
                    if triple:
                        triples.append(triple)
            
            # Effect relationships
            for pattern in self.effect_patterns:
                if pattern in sent_text:
                    triple = self._extract_effect_triple(sent, pattern)
                    if triple:
                        triples.append(triple)
            
            # Part-of relationships
            for pattern in self.part_of_patterns:
                if pattern in sent_text:
                    triple = self._extract_partof_triple(sent, pattern)
                    if triple:
                        triples.append(triple)
        
        return triples
    
    def _extract_entity_relations(self, doc) -> List[Tuple[str, str, str]]:
        triples = []
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Find entities that appear close to each other
        for i, (ent1, label1) in enumerate(entities):
            for ent2, label2 in entities[i+1:]:
                # Find verbs between entities
                ent1_idx = doc.text.find(ent1)
                ent2_idx = doc.text.find(ent2, ent1_idx + len(ent1))
                
                if 0 < ent2_idx - ent1_idx < 100:  # Within 100 chars
                    between_text = doc.text[ent1_idx:ent2_idx + len(ent2)]
                    between_doc = self.nlp(between_text)
                    
                    for token in between_doc:
                        if token.pos_ == "VERB":
                            triples.append((ent1, token.lemma_, ent2))
                            break
        
        return triples
    
    def _extract_causal_triple(self, sent, pattern: str) -> Tuple[str, str, str]:
        for token in sent:
            if token.lemma_ in self.causal_patterns:
                # Find subject before the causal verb
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                # Find object after the causal verb
                objects = [child for child in token.children if child.dep_ in ("dobj", "ccomp", "xcomp")]
                
                if subjects and objects:
                    subj_phrase = self._get_noun_phrase(subjects[0])
                    obj_phrase = self._get_noun_phrase(objects[0])
                    if subj_phrase and obj_phrase:
                        return (subj_phrase, "causes", obj_phrase)
        return None
    
    def _extract_effect_triple(self, sent, pattern: str) -> Tuple[str, str, str]:
        for token in sent:
            if token.lemma_ in self.effect_patterns:
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "attr")]
                
                if subjects and objects:
                    subj_phrase = self._get_noun_phrase(subjects[0])
                    obj_phrase = self._get_noun_phrase(objects[0])
                    if subj_phrase and obj_phrase:
                        return (subj_phrase, "affects", obj_phrase)
        return None
    
    def _extract_partof_triple(self, sent, pattern: str) -> Tuple[str, str, str]:
        for token in sent:
            if token.lemma_ in self.part_of_patterns:
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "attr", "prep")]
                
                if subjects and objects:
                    subj_phrase = self._get_noun_phrase(subjects[0])
                    obj_phrase = self._get_noun_phrase(objects[0])
                    if subj_phrase and obj_phrase:
                        return (subj_phrase, "part_of", obj_phrase)
        return None
    
    def _get_noun_phrase(self, token) -> str:
        # Get compound nouns and modifiers
        phrase_tokens = [token]
        
        # Add compounds
        for child in token.children:
            if child.dep_ in ("compound", "amod", "nmod"):
                phrase_tokens.insert(0, child)
        
        # Add right-side modifiers
        for child in token.children:
            if child.dep_ in ("prep", "relcl"):
                phrase_tokens.append(child)
        
        phrase = " ".join([t.text for t in sorted(phrase_tokens, key=lambda x: x.i)])
        return phrase.strip() if len(phrase.strip()) > 2 else None
    
    def _classify_nodes(
        self,
        nodes_with_context: List[Tuple],
        batch_size: int = 15
    ) -> Dict[str, str]:
        classifications = {}
        
        for node_name, predicates, chunk_types, contexts in tqdm(nodes_with_context, desc="Classifying nodes"):
            doc = self.nlp(node_name)
            
            # Check if it's a named entity
            if doc.ents:
                ent = doc.ents[0]
                if ent.label_ == "PERSON":
                    classifications[node_name] = "person"
                elif ent.label_ in ("ORG", "NORP"):
                    classifications[node_name] = "organization"
                elif ent.label_ in ("GPE", "LOC", "FAC"):
                    classifications[node_name] = "location"
                elif ent.label_ == "EVENT":
                    classifications[node_name] = "event"
                elif ent.label_ in ("DATE", "TIME"):
                    classifications[node_name] = "temporal"
                elif ent.label_ in ("PRODUCT", "WORK_OF_ART"):
                    classifications[node_name] = "product"
                elif ent.label_ in ("QUANTITY", "PERCENT", "MONEY", "CARDINAL"):
                    classifications[node_name] = "metric"
                else:
                    classifications[node_name] = "entity"
            else:
                # Classify based on POS tags and context
                main_token = doc[0] if len(doc) > 0 else None
                
                if main_token:
                    # Check chunk type for visual elements
                    if "image" in chunk_types:
                        classifications[node_name] = "visual_element"
                    # Check for process/action indicators
                    elif main_token.pos_ == "VERB" or any(word in node_name.lower() for word in ["process", "method", "procedure", "technique"]):
                        classifications[node_name] = "process"
                    # Check for technical/system terms
                    elif any(word in node_name.lower() for word in ["system", "technology", "tool", "software", "hardware"]):
                        classifications[node_name] = "technology"
                    # Check for data/attribute terms
                    elif any(word in node_name.lower() for word in ["data", "information", "record", "dataset"]):
                        classifications[node_name] = "data"
                    # Check predicates for attribute patterns
                    elif predicates and any(p in self.attribute_patterns for p in predicates):
                        classifications[node_name] = "attribute"
                    # Default based on POS
                    elif main_token.pos_ == "PROPN":
                        classifications[node_name] = "entity"
                    elif main_token.pos_ in ("NOUN", "PRON"):
                        classifications[node_name] = "concept"
                    else:
                        classifications[node_name] = "concept"
                else:
                    classifications[node_name] = "concept"
        
        return classifications
    
    def _detect_aliases(
        self,
        nodes: List[str],
        batch_size: int = 25
    ) -> Dict[str, str]:
        aliases_map = {}
        nodes_sorted = sorted(nodes, key=len, reverse=True)  # Process longer names first
        
        processed = set()
        groups = []  # List of sets of similar nodes
        
        for node in tqdm(nodes_sorted, desc="Detecting aliases"):
            if node in processed:
                continue
            
            # Create a new group for this node
            group = {node}
            processed.add(node)
            
            # Find similar nodes
            for other_node in nodes_sorted:
                if other_node in processed:
                    continue
                
                # Check string similarity
                similarity = self._calculate_similarity(node, other_node)
                
                if similarity > 0.8:  # High similarity threshold
                    group.add(other_node)
                    processed.add(other_node)
                # Check if one is substring of other (e.g., "CO2" vs "CO2 emissions")
                elif node.lower() in other_node.lower() or other_node.lower() in node.lower():
                    group.add(other_node)
                    processed.add(other_node)
                # Check for common abbreviations
                elif self._is_abbreviation(node, other_node):
                    group.add(other_node)
                    processed.add(other_node)
            
            if len(group) > 1:
                groups.append(group)
        
        # For each group, choose canonical form (most frequent or longest)
        for group in groups:
            # Choose longest form as canonical (usually most descriptive)
            canonical = max(group, key=len)
            for node in group:
                aliases_map[node] = canonical
        
        # Default mapping for unprocessed nodes
        for node in nodes:
            if node not in aliases_map:
                aliases_map[node] = node
        
        return aliases_map
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _is_abbreviation(self, short: str, long: str) -> bool:
        if len(short) >= len(long):
            return False
        
        # Check if short is acronym of long
        long_words = long.split()
        if len(short) == len(long_words):
            acronym = "".join([w[0] for w in long_words])
            if acronym.lower() == short.lower():
                return True
        
        # Check if short matches first letters
        long_clean = long.replace(" ", "").replace("-", "")
        if len(short) <= len(long_clean):
            if long_clean[:len(short)].lower() == short.lower():
                return True
        
        return False
    
    def _build_initial_graph(
        self,
        all_triple_contexts: List[Tuple]
    ):
        print("Building initial graph...")
        
        for (s, p, o), chunk_id, chunk_type in all_triple_contexts:
            s_meta = self.node_tracker.get_node_metadata(s)
            o_meta = self.node_tracker.get_node_metadata(o)
            
            s_canon = s_meta["canonical_name"]
            o_canon = o_meta["canonical_name"]
            
            # Add edge with source information
            if self.graph.has_edge(s_canon, o_canon):
                # Accumulate source chunks
                existing_sources = self.graph[s_canon][o_canon].get('source_chunks', [])
                self.graph[s_canon][o_canon]['source_chunks'] = existing_sources + [chunk_id]
            else:
                self.graph.add_edge(s_canon, o_canon, label=p, source_chunks=[chunk_id])
    
    def _save_consolidated_graph(self):
        """Save the consolidated graph to JSONL format"""
        print(f"Saving consolidated graph to {settings.KG_JSONL_PATH}...")
        
        with open(settings.KG_JSONL_PATH, "w", encoding="utf-8") as f:
            for source, target, data in self.graph.edges(data=True):
                s_meta = self.node_tracker.get_node_metadata(source)
                o_meta = self.node_tracker.get_node_metadata(target)
                
                entry = {
                    "subject": source,
                    "subject_canonical": s_meta["canonical_name"],
                    "subject_type": s_meta["type"],
                    "predicate": data.get('label', 'related_to'),
                    "object": target,
                    "object_canonical": o_meta["canonical_name"],
                    "object_type": o_meta["type"],
                    "weight": data.get('weight', 1),
                    "source_chunks": data.get('source_chunks', [])
                }
                
                # Add merged predicates if available
                if 'merged_predicates' in data:
                    entry['merged_predicates'] = data['merged_predicates']
                
                f.write(json.dumps(entry) + "\n")
        
        print(f"Saved {self.graph.number_of_edges()} edges to {settings.KG_JSONL_PATH}")
