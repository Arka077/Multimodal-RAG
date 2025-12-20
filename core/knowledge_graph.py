"""
Knowledge graph extraction and management using rule-based methods
"""
import re
import json
import networkx as nx
import spacy
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Set
from pathlib import Path
from tqdm.auto import tqdm
from difflib import SequenceMatcher

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


class KnowledgeGraphBuilder:
    """Build knowledge graph from processed documents using rule-based methods"""
    
    def __init__(self):
        print("Loading SpaCy model for knowledge graph extraction...")
        self.nlp = spacy.load(settings.SPACY_MODEL)
        self.node_tracker = NodeMetadataTracker()
        self.graph = nx.DiGraph()
        
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
        chunks: List[Dict[str, Any]]
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
        
        # Stage 3: Detect aliases
        aliases_map = self._detect_aliases(list(self.node_tracker.node_data.keys()))
        
        for original, canonical in aliases_map.items():
            if original in self.node_tracker.node_data:
                self.node_tracker.node_data[original]["canonical_name"] = canonical
        
        # Stage 4: Build final graph
        self._build_final_graph(all_triple_contexts)
        
        # Get metadata
        node_metadata = {
            n: self.node_tracker.get_node_metadata(n)
            for n in self.node_tracker.node_data
        }
        
        print(f"Knowledge graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph, node_metadata
    
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
    
    def _build_final_graph(
        self,
        all_triple_contexts: List[Tuple]
    ):
        print("Building final graph...")
        
        with open(settings.KG_JSONL_PATH, "w", encoding="utf-8") as f:
            for (s, p, o), chunk_id, chunk_type in all_triple_contexts:
                s_meta = self.node_tracker.get_node_metadata(s)
                o_meta = self.node_tracker.get_node_metadata(o)
                
                s_canon = s_meta["canonical_name"]
                o_canon = o_meta["canonical_name"]
                
                self.graph.add_edge(s_canon, o_canon, label=p)
                
                f.write(json.dumps({
                    "subject": s,
                    "subject_canonical": s_canon,
                    "subject_type": s_meta["type"],
                    "predicate": p,
                    "object": o,
                    "object_canonical": o_canon,
                    "object_type": o_meta["type"],
                    "source_chunk": chunk_id
                }) + "\n")
