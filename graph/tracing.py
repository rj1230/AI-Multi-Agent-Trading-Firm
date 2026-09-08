"""
Per-node trace logging: every node execution is wrapped with a trace_id
linking it to a full run, logging input state, output state, latency, and
(for LLM nodes) token usage. Makes every trade decision replayable.
"""
import time
from contextlib import contextmanager


@contextmanager
def traced_node(node_name: str, state: dict):
    """Usage: with traced_node('news_agent', state): ... """
    start = time.time()
    try:
        yield
    finally:
        latency = time.time() - start
        # TODO: append {node_name, trace_id, latency, ...} to state['agent_logs']
        raise NotImplementedError
