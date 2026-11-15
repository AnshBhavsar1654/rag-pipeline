"""
Graph DB — knowledge graph of entities and relationships

While indexing, an LLM reads the chunks in batches and extracts
(entity)-[RELATION]->(entity) triples. They are kept in a NetworkX graph
persisted as JSON and queried with Cypher (via grand-cypher), so questions
about how things are connected can be answered by graph traversal.

Node ids and ``name`` attributes are lower-case so Cypher matches are not
case sensitive; ``__labels__`` holds the node type / relationship type,
which is what grand-cypher matches ``(n:Type)`` and ``[r:TYPE]`` against.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def _norm_name(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def _norm_label(text: str, upper: bool) -> str:
    label = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_") or "Entity"
    if label[0].isdigit():
        label = f"_{label}"
    return label.upper() if upper else label[0].upper() + label[1:]


class GraphStore:
    """A persisted directed multigraph of extracted triples."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.graph = self._load()

    def _load(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        if not self.path.exists():
            return graph
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            graph.add_node(node["id"], name=node["id"], __labels__=set(node.get("labels", ["Entity"])))
        for edge in data.get("edges", []):
            graph.add_edge(
                edge["head"], edge["tail"],
                __labels__={edge["relation"]}, relation=edge["relation"],
                source=edge.get("source", ""),
            )
        return graph

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {"id": n, "labels": sorted(a.get("__labels__", {"Entity"}))}
                for n, a in self.graph.nodes(data=True)
            ],
            "edges": [
                {"head": h, "tail": t, "relation": a["relation"], "source": a.get("source", "")}
                for h, t, a in self.graph.edges(data=True)
            ],
        }
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    # -- writing ------------------------------------------------------------

    def add_triples(self, triples: list[dict], source: str) -> int:
        added = 0
        for t in triples:
            head, tail = _norm_name(t.get("head", "")), _norm_name(t.get("tail", ""))
            relation = _norm_label(t.get("relation", ""), upper=True)
            if not head or not tail or head == tail:
                continue
            for name, kind in ((head, t.get("head_type")), (tail, t.get("tail_type"))):
                label = _norm_label(kind or "Entity", upper=False)
                if name in self.graph:
                    labels = self.graph.nodes[name]["__labels__"]
                    if label != "Entity":
                        labels.discard("Entity")
                        labels.add(label)
                else:
                    self.graph.add_node(name, name=name, __labels__={label})
            exists = any(
                a["relation"] == relation
                for a in (self.graph.get_edge_data(head, tail) or {}).values()
            )
            if not exists:
                self.graph.add_edge(head, tail, __labels__={relation}, relation=relation, source=source)
                added += 1
        return added

    def delete_sources(self, sources: set[str]) -> None:
        stale = [
            (h, t, k) for h, t, k, a in self.graph.edges(keys=True, data=True)
            if a.get("source") in sources
        ]
        self.graph.remove_edges_from(stale)
        self.graph.remove_nodes_from([n for n in list(self.graph) if self.graph.degree(n) == 0])

    def reset(self) -> None:
        self.graph = nx.MultiDiGraph()
        if self.path.exists():
            self.path.unlink()

    # -- reading ------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return self.graph.number_of_edges() == 0

    def schema(self, max_entities: int = 60) -> str:
        """Node labels, relationship types and example entities, for Text-to-Cypher."""
        if self.is_empty:
            return ""
        labels = sorted({l for _, a in self.graph.nodes(data=True) for l in a["__labels__"]})
        relations = sorted({a["relation"] for _, _, a in self.graph.edges(data=True)})
        patterns = sorted({
            f"(:{sorted(self.graph.nodes[h]['__labels__'])[0]})-[:{a['relation']}]->"
            f"(:{sorted(self.graph.nodes[t]['__labels__'])[0]})"
            for h, t, a in self.graph.edges(data=True)
        })
        busiest = sorted(self.graph.nodes, key=self.graph.degree, reverse=True)[:max_entities]
        return (
            f"Node labels: {', '.join(labels)}\n"
            f"Relationship types: {', '.join(relations)}\n"
            f"Patterns:\n" + "\n".join(patterns[:80]) + "\n"
            f"Example entity names (all lower-case): {', '.join(busiest)}"
        )

    def run_cypher(self, query: str) -> list[dict]:
        """Run a Cypher query; returns one dict per result row."""
        from grandcypher import GrandCypher

        result = GrandCypher(self.graph).run(query)
        columns = list(result.keys())
        length = max((len(v) for v in result.values()), default=0)
        rows = []
        for i in range(length):
            rows.append({str(c): _plain(result[c][i]) if i < len(result[c]) else None for c in columns})
        return rows

    def neighborhood(self, text: str, limit: int = 40) -> list[str]:
        """Facts about every known entity mentioned in *text* (Cypher fallback)."""
        lowered = f" {_norm_name(text)} "
        mentioned = [
            n for n in self.graph.nodes
            if len(n) > 2 and re.search(rf"(?<![0-9a-z]){re.escape(n)}(?![0-9a-z])", lowered)
        ]
        facts: list[str] = []
        for node in sorted(mentioned, key=len, reverse=True):
            for h, t, a in list(self.graph.out_edges(node, data=True)) + list(self.graph.in_edges(node, data=True)):
                fact = f"({h}) -[{a['relation']}]-> ({t})"
                if fact not in facts:
                    facts.append(fact)
        return facts[:limit]


def _plain(value):
    """Make grand-cypher values (sets, dicts of edges) printable."""
    if isinstance(value, set):
        return ", ".join(sorted(map(str, value)))
    if isinstance(value, dict):
        labels = [
            ", ".join(sorted(v.get("__labels__", []))) for v in value.values() if isinstance(v, dict)
        ]
        return "; ".join(l for l in labels if l) or str(value)
    return value


# ---------------------------------------------------------------------------
# Triple extraction (indexing side of the graph DB)
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = ChatPromptTemplate.from_template("""\
Extract a knowledge graph from the passages below. Return a JSON array of \
triples, each an object with keys "head", "head_type", "relation", "tail", \
"tail_type". Use short canonical entity names (e.g. "GPT-4", "OpenAI"), \
singular CamelCase types (e.g. Organization, Model, Concept, Person) and \
UPPER_SNAKE_CASE relations (e.g. DEVELOPED_BY, PART_OF, USES). Only include \
facts stated in the passages, at most 25 of the most important ones. Output ONLY the JSON array.

Passages:
{passages}""")


def parse_triples(text: str) -> list[dict]:
    """Parse triples one object at a time, so a reply cut off at the token
    limit still yields every complete triple before the cut."""
    triples = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            item = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("head") and item.get("tail"):
            triples.append(item)
    return triples


class GraphExtractor:
    """Batch chunks through an LLM to pull out entity-relationship triples."""

    def __init__(self, llm: BaseChatModel, batch_size: int = 3):
        self._chain = _EXTRACT_PROMPT | llm | StrOutputParser()
        self._batch_size = max(1, batch_size)

    def build(self, store: GraphStore, chunks: list[Document]) -> int:
        by_source: dict[str, list[Document]] = {}
        for chunk in chunks:
            by_source.setdefault(str(chunk.metadata.get("source", "")), []).append(chunk)

        added = 0
        batches = sum(-(-len(c) // self._batch_size) for c in by_source.values())
        done = 0
        for source, docs in by_source.items():
            for i in range(0, len(docs), self._batch_size):
                batch = docs[i: i + self._batch_size]
                passages = "\n\n".join(f"[{j + 1}] {d.page_content}" for j, d in enumerate(batch))
                done += 1
                try:
                    triples = parse_triples(self._chain.invoke({"passages": passages}))
                except Exception as e:
                    print(f"[WARN]  Graph extraction failed for a batch of {Path(source).name}: {e}")
                    continue
                added += store.add_triples(triples, source)
                print(f"[GRAPH] Batch {done}/{batches}: {len(triples)} triple(s)")
        return added
