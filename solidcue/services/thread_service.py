from uuid import uuid4


def create_thread_id() -> str:
    """Create a new LangGraph thread id for checkpointed runs."""
    return str(uuid4())
