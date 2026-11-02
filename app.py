import json
import re
from pathlib import Path

import streamlit as st

from my_agent.Coordinator_agent import run_coordinator
from my_agent.citations import format_answer_citations
from my_agent.permission import load_roles


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Project Management Agent",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.25);
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        color: #777;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .online-status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 12px;
        border-radius: 20px;
        background: rgba(40, 167, 69, 0.10);
        color: #28a745;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    .online-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #28a745;
    }

    .user-card {
        padding: 14px;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .user-name {
        font-weight: 600;
        font-size: 1rem;
    }

    .user-role {
        font-size: 0.82rem;
        color: #777;
        margin-top: 3px;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #777;
        margin-bottom: 8px;
    }

    .info-note {
        color: #777;
        font-size: 0.82rem;
        margin-top: 4px;
        margin-bottom: 0;
    }

    .permission-allowed {
        color: #28a745;
        font-weight: 600;
    }

    .permission-denied {
        color: #999;
        font-weight: 500;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.75rem;
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# USER DATA
# ==================================================

USERS_FILE = Path(__file__).resolve().parent / "Memory" / "users.json"

try:
    with open(USERS_FILE, encoding="utf-8") as users_file:
        users_data = json.load(users_file)
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Could not load users from {USERS_FILE}: {error}")
    st.stop()

if not isinstance(users_data, list) or any(
    not isinstance(user, dict)
    or not isinstance(user.get("user_id"), str)
    or not user["user_id"].strip()
    or not isinstance(user.get("name"), str)
    or not user["name"].strip()
    for user in users_data
):
    st.error(
        "Expected users.json to contain user records with user_id and name: "
        f"{USERS_FILE}"
    )
    st.stop()

active_users = [
    user
    for user in users_data
    if str(user.get("status", "")).casefold() == "active"
]
users = {
    f"{user['name'].strip()} ({user['user_id'].strip()})": user
    for user in active_users
}

if not users:
    st.error(f"No active users are configured in {USERS_FILE}.")
    st.stop()

permission_labels = {
    "VIEW_PROJECT": "View Projects",
    "VIEW_SPACE": "View Spaces",
    "VIEW_SECTION": "View Sections",
    "VIEW_TASK": "View Tasks",
    "VIEW_EMPLOYEE": "View Employees",
    "VIEW_KNOWLEDGE_BASE": "Search Project Issues",
}


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_recent_conversation(messages, max_messages=8, max_chars=6000):
    """Return recent user/assistant text messages within a size limit."""
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ][-max_messages:]

    while history and sum(len(message["content"]) for message in history) > max_chars:
        history.pop(0)

    return history

# ==================================================
# EXECUTION DETAILS
# ==================================================

def render_execution_details(
    trace,
    iterations,
    memory_context=None
):
    with st.expander("🔍 Execution Details"):

        st.caption(
            f"Coordinator iterations: {iterations}"
        )

        if memory_context:
            st.markdown(
                "🧠 **Memory Context Used**"
            )
            st.caption(
                "Relevant user memory was retrieved for this request."
            )

        if not trace:
            st.caption("No execution trace available.")

        else:
            for step_index, step in enumerate(trace, start=1):

                agent = step.get(
                    "agent",
                    "Unknown Agent"
                )

                agent_result = step.get(
                    "result",
                    {}
                )

                st.markdown(
                    f"**{step_index}. Coordinator → {agent}**"
                )

                if isinstance(agent_result, dict):

                    specialist_agent = agent_result.get("agent")

                    if specialist_agent:
                        st.markdown(
                            f"↳ 🤖 **{specialist_agent}**"
                        )

                    specialist_trace = agent_result.get(
                        "trace",
                        []
                    )

                    for tool_step in specialist_trace:

                        tool_name = tool_step.get(
                            "tool",
                            "Unknown Tool"
                        )

                        tool_arguments = tool_step.get(
                            "arguments",
                            {}
                        )

                        tool_result = tool_step.get(
                            "result",
                            {}
                        )

                        st.markdown(
                            f"↳ 🔧 `{tool_name}`"
                        )

                        if isinstance(tool_arguments, dict):
                            for key, value in tool_arguments.items():
                                st.caption(
                                    f"{key.replace('_', ' ').title()}: {value}"
                                )

                        if isinstance(tool_result, dict):

                            if tool_result.get("success") is False:
                                st.error(
                                    "✗ Tool execution failed"
                                )
                            else:
                                st.success(
                                    "✓ Completed",
                                    icon="✅"
                                )

                        else:
                            st.success(
                                "✓ Completed",
                                icon="✅"
                            )

                if step_index < len(trace):
                    st.divider()

            st.divider()

            has_tool_failure = any(
                isinstance(specialist_step.get("result"), dict)
                and any(
                    isinstance(tool_step.get("result"), dict)
                    and tool_step.get("result", {}).get("success") is False
                    for tool_step in specialist_step.get("result", {}).get("trace", [])
                )
                for specialist_step in trace
            )

            if has_tool_failure:
                st.warning(
                    "Execution completed with tool errors.",
                    icon="⚠️"
                )
            else:
                st.success(
                    "Execution completed successfully.",
                    icon="🤖"
                )

#==================================================
# Render Citations
#==================================================

def render_citations(answer, citations):
    """Display sources referenced by citation markers in the answer."""
    cited_ids = set(re.findall(r"\[(S\d+)\]", answer or ""))

    cited_sources = [
        citation
        for citation in citations or []
        if isinstance(citation, dict)
        and citation.get("id") in cited_ids
    ]

    if not cited_sources:
        return

    with st.expander("📚 Sources"):
        for citation in cited_sources:
            st.text(
                f"[{citation['id']}] {citation.get('label', 'Source')}"
            )
            st.caption(
                f"{citation.get('source', '')} · "
                f"{citation.get('record_id', '')}"
            )

            passage = citation.get("passage")
            if passage:
                st.text(passage)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🤖 Project Agent")

    st.markdown(
        '<div class="section-label">Current User</div>',
        unsafe_allow_html=True
    )

    selected_user = st.selectbox(
        "Select User",
        options=list(users.keys()),
        label_visibility="collapsed"
    )

    selected_user_data = users[selected_user]
    user_id = selected_user_data["user_id"].strip()
    user_name = selected_user_data["name"].strip()
    user_role = str(selected_user_data.get("role") or "No role assigned")

    st.markdown(
        f"""
        <div class="user-card">
            <div class="user-name">👤 {user_name}</div>
            <div class="user-role">{user_role}</div>
            <div class="user-role">ID: {user_id}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()


    st.markdown(
        '<div class="section-label">Available Users</div>',
        unsafe_allow_html=True
    )

    for user in active_users:
        st.caption(
            f"**{user['user_id'].strip()}** · {user['name'].strip()}"
        )

    st.divider()

    st.markdown(
        """
        <div class="online-status">
            <span class="online-dot"></span>
            System Online
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# ==================================================
# MAIN HEADER
# ==================================================

st.markdown(
    '<div class="main-title">Project Management Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">Multi-Agent AI Project Management System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="online-status">
        <span class="online-dot"></span>
        Assistant Ready
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HOW TO USE & PERMISSIONS
# ==================================================

col1, col2 = st.columns(2)

with col1:

    with st.expander("📖 How to use"):

        st.markdown(
            """
            Ask questions in natural language. The system routes your request
            to the appropriate specialist agent.

            **Available features:**

            - 📁 **Workspaces (projects)** — look up workspaces by name or ID;
              ask about status, dates, descriptions, and available stakeholder fields.
            - 🗂️ **Spaces and sections** — list or filter them by name, ID,
              status, or their parent workspace.
            - 📋 **Tasks** — find tasks by name or exact task ID; ask about
              status, assignments, due dates, and overdue tasks.
            - 👥 **Employees** — look up available Resource records by ID,
              name, or department. Access depends on your permissions.
            - 🐞 **Project issues** — search issue titles, descriptions,
              reproducible steps, and IDs from the issue records.

            **Try asking:**
            - *Which workspaces are currently on hold?*
            - *Show the spaces in the workspace Corporate Head Quarters.*
            - *Look up a task by its exact ID.*
            - *Find an employee by name or department.*
            - *Find the issue about the 2024-25 audit report.*
            """
        )

with col2:

    with st.expander("🔐 Your permissions"):

        all_roles = load_roles()
        current_permissions = set(
            all_roles.get(user_id, {}).get("permissions", [])
        )

        st.markdown(
            f"**Selected user:** {user_name} (`{user_id}`)"
        )
        st.caption(
            f"Configured role: {user_role} · "
            f"Department: {selected_user_data.get('department') or 'Not specified'}"
        )
        st.markdown("**Access granted to this user:**")

        for permission, label in permission_labels.items():
            allowed = permission in current_permissions

            if allowed:
                st.markdown(
                    f'<div class="permission-allowed">✓ {label}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="permission-denied">✗ {label}</div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            '<p class="info-note">Access is read from the selected user’s role permissions. Employee lookup is unavailable when View Employees is not granted.</p>',
            unsafe_allow_html=True
        )


# ==================================================
# DISPLAY CONVERSATION HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            format_answer_citations(
                message["content"],
                message.get("citations", []),
            )
        )

        if message["role"] == "assistant":
            render_citations(
                message["content"],
                message.get("citations", []),
            )

            if "trace" in message:
                render_execution_details(
                    message["trace"],
                    message.get("iterations", 0),
                    message.get("memory_context")
                )

# ==================================================
# EMPTY STATE
# ==================================================

if not st.session_state.messages:

    st.info(
        "👋 Ask me about projects, tasks, employees, project progress, or issues."
    )


# ==================================================
# CHAT INPUT
# ==================================================

query = st.chat_input(
    "Ask about projects, tasks, employees, or issues..."
)


# ==================================================
# HANDLE NEW MESSAGE
# ==================================================

if query:

    conversation_history = get_recent_conversation(
        st.session_state.messages
    )

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # ==================================================
    # RUN REAL COORDINATOR
    # ==================================================

    with st.chat_message("assistant"):

        try:

            with st.spinner("🤖 Coordinator is working..."):

                result = run_coordinator(
                    user_id=user_id,
                    user_query=query,
                    conversation_history = conversation_history
                )

            # ------------------------------------------
            # VALIDATE RESULT
            # ------------------------------------------

            if not isinstance(result, dict):

                response = str(result)

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            # ------------------------------------------
            # SUCCESSFUL RESPONSE
            # ------------------------------------------

            elif result.get("success"):

                response = result.get("report")

                if not response:
                    response = (
                        "The request was completed, "
                        "but no report was returned."
                    )

                trace = result.get(
                    "trace",
                    []
                )

                iterations = result.get(
                    "iterations",
                    0
                )

                memory_context = result.get(
                    "memory_context",
                    {}
                )

                citations = result.get("citations", [])

                st.markdown(
                    format_answer_citations(response, citations)
                )

                render_citations(response, citations)

                render_execution_details(
                    trace,
                    iterations,
                    memory_context
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "trace": trace,
                    "iterations": iterations,
                    "memory_context": memory_context,
                    "citations": citations,
                })

            # ------------------------------------------
            # COORDINATOR RETURNED FAILURE
            # ------------------------------------------

            else:

                response = result.get(
                    "report",
                    "The Coordinator could not complete the request."
                )

                citations = result.get("citations", [])

                st.error(
                    format_answer_citations(response, citations)
                )
                render_citations(response, citations)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "citations": citations,
                })

        # ----------------------------------------------
        # BACKEND ERROR
        # ----------------------------------------------

        except Exception as error:

            st.error(
                "The request could not be processed."
            )

            with st.expander("Technical Details"):

                st.code(
                    str(error),
                    language="text"
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "Sorry, I couldn't process that request."
                )
            })


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        Project Management Agent · Multi-Agent AI System
    </div>
    """,
    unsafe_allow_html=True
)