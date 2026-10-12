# reverse_edges.py
#
# Full edge reversal for LightRAG ablation study.
#
# Reverses ALL aspects of edge directionality:
#   1. graph_chunk_entity_relation.graphml:
#        - Swaps source/target on every edge
#        - Swaps entity name occurrences in edge descriptions
#   2. vdb_relationships.json:
#        - Swaps src_id/tgt_id
#        - Swaps entity names in content (header + description)
#        - Optionally recomputes embeddings (--recompute_embeddings)
#
# Produces nonsensical descriptions by design, to test LightRAG's
# robustness to fully reversed edges.
#
# Usage:
#   python reverse_edges.py --workspace ./lightrag_workspace/corpus_name
#   python reverse_edges.py --workspace ./lightrag_workspace/corpus_name --recompute_embeddings
#
# Output files sit alongside originals with _reversed suffix.
# Rename them to replace originals; keep originals to restore baseline.

import json
import re
import argparse
import asyncio
import base64
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Core text utility
# ---------------------------------------------------------------------------

def swap_names_in_text(text: str, src: str, tgt: str) -> str:
    """
    Swap all occurrences of src and tgt entity names in text
    (case-insensitive). Replaces the longer name first via a placeholder
    to avoid double-replacement when one name is a substring of the other
    (e.g. 'Skin Cancer' inside 'Basal Cell Skin Cancer').
    """
    if not text:
        return text
    PLACEHOLDER = "\x00PLACEHOLDER\x00"
    longer, shorter = (src, tgt) if len(src) >= len(tgt) else (tgt, src)
    # Step 1: longer name → placeholder
    text = re.sub(re.escape(longer), PLACEHOLDER, text, flags=re.IGNORECASE)
    # Step 2: shorter name → longer name
    text = re.sub(re.escape(shorter), longer, text, flags=re.IGNORECASE)
    # Step 3: placeholder → shorter name
    text = text.replace(PLACEHOLDER, shorter)
    return text


# ---------------------------------------------------------------------------
# GraphML reversal
# ---------------------------------------------------------------------------

def reverse_graphml(graphml_path: Path, output_path: Path) -> int:
    """
    Reverse all edges in the graphml:
      - swap source/target attributes
      - swap entity names in description (key d5)
    Returns number of edges reversed.
    """
    # Preserve namespace declarations
    ET.register_namespace('', 'http://graphml.graphdrawing.org/xmlns')
    ET.register_namespace(
        'xsi', 'http://www.w3.org/2001/XMLSchema-instance'
    )

    tree = ET.parse(graphml_path)
    root = tree.getroot()
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    graph = root.find('g:graph', ns)
    edges = graph.findall('g:edge', ns)

    DESC_KEY = 'd5'
    for edge in edges:
        src = edge.get('source')
        tgt = edge.get('target')

        # Swap structural direction
        edge.set('source', tgt)
        edge.set('target', src)

        # Swap entity names in description
        for data in edge.findall('g:data', ns):
            if data.get('key') == DESC_KEY and data.text:
                data.text = swap_names_in_text(data.text, src, tgt)

    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    return len(edges)


# ---------------------------------------------------------------------------
# VDB relationships reversal
# ---------------------------------------------------------------------------

def reverse_vdb_relationships(
    vdb_path: Path,
    output_path: Path
) -> int:
    """
    Reverse all relationships in vdb_relationships.json:
      - swap src_id/tgt_id
      - swap entity names in content (header line + description lines)
    Returns number of entries reversed.
    """
    with open(vdb_path, 'r', encoding='utf-8') as f:
        vdb = json.load(f)

    for entry in vdb['data']:
        src = entry.get('src_id', '')
        tgt = entry.get('tgt_id', '')

        # Swap metadata
        entry['src_id'] = tgt
        entry['tgt_id'] = src

        # Swap entity names throughout the content field
        if 'content' in entry and entry['content']:
            entry['content'] = swap_names_in_text(entry['content'], src, tgt)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vdb, f, ensure_ascii=False)

    return len(vdb['data'])


# ---------------------------------------------------------------------------
# Optional: recompute embeddings from reversed content
# ---------------------------------------------------------------------------

async def recompute_embeddings(
    vdb_path: Path,
    embed_model: str,
    ollama_host: str,
    batch_size: int = 16
) -> None:
    """
    Recompute embedding matrix from reversed content fields and
    write back to the same file. Requires Ollama running with the
    specified embedding model.
    """
    from lightrag.llm.ollama import ollama_embed

    with open(vdb_path, 'r', encoding='utf-8') as f:
        vdb = json.load(f)

    data = vdb['data']
    contents = [entry['content'] for entry in data]
    total = len(contents)
    all_embeddings = []

    print(f"  Recomputing {total} embeddings in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        batch = contents[i:i + batch_size]
        embeddings = await ollama_embed(
            batch, embed_model=embed_model, host=ollama_host
        )
        all_embeddings.extend(embeddings)
        print(
            f"  Embedded {min(i + batch_size, total)}/{total} "
            f"({min(i + batch_size, total)/total*100:.1f}%)"
        )

    matrix = np.array(all_embeddings, dtype=np.float32)
    vdb['matrix'] = base64.b64encode(matrix.tobytes()).decode('utf-8')
    vdb['embedding_dim'] = matrix.shape[1]

    with open(vdb_path, 'w', encoding='utf-8') as f:
        json.dump(vdb, f, ensure_ascii=False)

    print(f"  ✅ Recomputed embedding matrix: {matrix.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fully reverse all edge directions in a LightRAG workspace "
            "for ablation study, including descriptions."
        )
    )
    parser.add_argument(
        "--workspace", required=True,
        help="Path to LightRAG corpus workspace directory"
    )
    parser.add_argument(
        "--suffix", default="_reversed",
        help="Suffix for output filenames (default: _reversed)"
    )
    parser.add_argument(
        "--recompute_embeddings", action="store_true",
        help="Recompute vdb embeddings from reversed content (requires Ollama)"
    )
    parser.add_argument(
        "--embed_model", default="qwen3-embedding:0.6b",
        help="Ollama embedding model for recomputation"
    )
    parser.add_argument(
        "--ollama_host", default="http://localhost:11434",
        help="Ollama host URL"
    )
    args = parser.parse_args()

    workspace = Path(args.workspace)
    graphml_path = workspace / "graph_chunk_entity_relation_og.graphml"
    vdb_rel_path = workspace / "vdb_relationships_og.json"
    graphml_out  = workspace / f"graph_chunk_entity_relation{args.suffix}.graphml"
    vdb_rel_out  = workspace / f"vdb_relationships{args.suffix}.json"

    for p in [graphml_path, vdb_rel_path]:
        if not p.exists():
            print(f"❌ File not found: {p}")
            return

    print(f"\n📂 Workspace: {workspace}\n")

    # 1. Reverse graphml
    n = reverse_graphml(graphml_path, graphml_out)
    print(
        f"✅ graphml: reversed {n} edges "
        f"(source↔target + descriptions) → {graphml_out.name}"
    )

    # 2. Reverse vdb_relationships
    n = reverse_vdb_relationships(vdb_rel_path, vdb_rel_out)
    print(
        f"✅ vdb_relationships: reversed {n} entries "
        f"(src_id↔tgt_id + content) → {vdb_rel_out.name}"
    )

    # 3. Optionally recompute embeddings
    if args.recompute_embeddings:
        print(f"\n🔄 Recomputing embeddings using {args.embed_model}...")
        asyncio.run(recompute_embeddings(
            vdb_rel_out,
            args.embed_model,
            args.ollama_host
        ))
    else:
        print(
            "\n⚠️  Embeddings not recomputed (original vectors retained)."
            "\n   Add --recompute_embeddings to fully update the vector index."
            "\n   Practically, the effect is minimal since descriptions drive"
            "\n   semantic similarity more than the entity name header."
        )

    print(f"""
✅ Done. To use reversed files in your experiment:
   Rename {graphml_out.name} → graph_chunk_entity_relation.graphml
   Rename {vdb_rel_out.name}  → vdb_relationships.json

   Keep originals intact to restore baseline condition.
""")


if __name__ == "__main__":
    main()
