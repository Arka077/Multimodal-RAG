"""
Visualization utilities for knowledge graph and embeddings
"""
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from pyvis.network import Network
from typing import Dict, Any, Tuple

from config.settings import settings


class GraphVisualizer:
    """Visualize knowledge graph"""
    
    def __init__(self, graph: nx.DiGraph, node_metadata: Dict[str, Any]):
        self.graph = graph
        self.node_metadata = node_metadata
    
    def generate_visualization(self) -> Tuple[str, str]:
        if self.graph.number_of_nodes() == 0:
            return "<h3>Knowledge graph is empty. Add files to build it.</h3>", ""
        
        # Create PyVis network
        pyvis_graph = Network(
            height='750px',
            width='100%',
            notebook=True,
            directed=True,
            bgcolor="#222222",
            font_color="white"
        )
        
        # Color scheme by type
        type_colors = {
            "person": "#E74C3C",
            "organization": "#3498DB",
            "location": "#2ECC71",
            "concept": "#9B59B6",
            "technology": "#1ABC9C",
            "process": "#F39C12",
            "visual_element": "#FF6B6B",
            "data": "#F1C40F",
            "entity": "#7F8C8D"
        }
        
        # Map canonical names to metadata
        canonical_to_meta = {
            meta['canonical_name']: meta
            for meta in self.node_metadata.values()
        }
        
        # Add nodes
        for node in self.graph.nodes():
            meta = canonical_to_meta.get(node)
            if not meta:
                continue
            
            node_type = meta["type"]
            size = 15 + min(meta.get("frequency", 1) * 2, 30)
            color = type_colors.get(node_type, "#7F8C8D")
            border_color = "#FFD700" if meta["has_visual_representation"] else color
            
            tooltip = (
                f"<b>{node}</b><br>"
                f"Type: {node_type}<br>"
                f"Freq: {meta['frequency']}<br>"
                f"Sources: {len(meta['source_chunks'])}"
            )
            
            pyvis_graph.add_node(
                node,
                label=node,
                title=tooltip,
                color={"background": color, "border": border_color},
                size=size,
                borderWidth=3 if meta["has_visual_representation"] else 2
            )
        
        # Add edges
        for u, v, data in self.graph.edges(data=True):
            pyvis_graph.add_edge(
                u, v,
                label=data.get("label", ""),
                arrows="to",
                smooth={"type": "curvedCW", "roundness": 0.2}
            )
        
        # Generate HTML
        html = pyvis_graph.generate_html().replace("'", '"')
        iframe = f'<iframe style="width:100%; height:800px; border:2px solid #444;" srcdoc=\'{html}\'></iframe>'
        
        stats = f"**Nodes:** {self.graph.number_of_nodes()}\n**Edges:** {self.graph.number_of_edges()}"
        
        return iframe, stats


class EmbeddingVisualizer:
    """Visualize document embeddings in 3D space"""
    
    def __init__(self, faiss_index, chunk_ids: list, chunk_lookup: Dict):
        self.faiss_index = faiss_index
        self.chunk_ids = chunk_ids
        self.chunk_lookup = chunk_lookup
    
    def generate_visualization(self):
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return None
        
        print("Generating embedding visualization...")
        
        # Get embeddings
        embeddings = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
        
        # Dimensionality reduction
        n_neighbors = min(15, embeddings.shape[0] - 1)
        if n_neighbors <= 1:
            return None
        
        # Use UMAP if available, else use PCA
        if UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=3,
                metric="cosine",
                random_state=42
            )
            emb_3d = reducer.fit_transform(embeddings)
            reduction_method = "UMAP"
        else:
            # Fallback to PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3, random_state=42)
            emb_3d = pca.fit_transform(embeddings)
            reduction_method = "PCA"
        
        # Create DataFrame
        df = pd.DataFrame(emb_3d, columns=["x", "y", "z"])
        df["label"] = [
            self.chunk_lookup.get(cid, {}).get('short_summary', 'N/A')
            for cid in self.chunk_ids
        ]
        df["type"] = [
            self.chunk_lookup.get(cid, {}).get('type', 'N/A')
            for cid in self.chunk_ids
        ]
        
        # Create plot
        fig = px.scatter_3d(
            df,
            x="x", y="y", z="z",
            hover_name="label",
            color="type",
            title=f"3D Visualization of Chunk Embeddings ({reduction_method})",
            opacity=0.8
        )
        
        fig.update_traces(marker=dict(size=4))
        
        return fig
