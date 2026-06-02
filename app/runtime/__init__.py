"""Runtime semantic-retrieval layer.

Turns a natural-language clinical question into a narrowed semantic subgraph:
build a compact context from the indexed registry, let the LLM select relevant
root resources, then deterministically narrow the graph. No SQL is generated here.
"""
