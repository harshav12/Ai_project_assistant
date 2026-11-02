"""Temporary no-op memory tools while memory data is test-only."""

def get_active_memories(user_id):
    """Return no memories while persistent memory is disabled."""
    return []


def get_relevant_memories(user_id, project_id=None, task_id=None):
    """Return no memories while persistent memory is disabled."""
    return []


def get_relevant_memories_for_query(user_id, user_query):
    """Return no memories while persistent memory is disabled."""
    return []


def find_project_id_in_query(user_query):
    """Return no project ID while persistent memory is disabled."""
    return None


def get_conversation_context(user_id, user_query):
    """Return no conversation memories while persistent memory is disabled."""
    return []


def get_memory_context(user_id, user_query):
    """Build an empty memory context while persistent memory is disabled."""
    return {"user_id": user_id, "memories": []}


def update_preference(
    user_id,
    project_id,
    value,
    status="active",
):
    """Do not persist preference updates while persistent memory is disabled."""
    return None


def process_preference_update(user_id, user_query):
    """Do not process preference updates while persistent memory is disabled."""
    return None