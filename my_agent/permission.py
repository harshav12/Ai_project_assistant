import json
from enum import Enum
from pathlib import Path


class Permission(str, Enum):
    VIEW_PROJECT = "VIEW_PROJECT"
    VIEW_SPACE = "VIEW_SPACE"
    VIEW_SECTION = "VIEW_SECTION"
    VIEW_TASK = "VIEW_TASK"
    VIEW_EMPLOYEE = "VIEW_EMPLOYEE"
    VIEW_KNOWLEDGE_BASE = "VIEW_KNOWLEDGE_BASE"


BASE_DIR = Path(__file__).resolve().parent

ROLES_FILE = BASE_DIR / "roles.json"


def load_roles():
    with open(ROLES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def has_permission(user_id, permission):
    roles = load_roles()
    user_data = roles.get(user_id)

    if user_data is None:
        return False

    permissions = user_data.get("permissions", [])

    return permission.value in permissions


def require_permission(user_id, permission):
    if not has_permission(user_id, permission):
        raise PermissionError(
            f"User {user_id} is not authorized to perform "
            f"'{permission.value}'."
        )

    return True


TOOL_PERMISSIONS = {
    "get_projects": Permission.VIEW_PROJECT,
    "get_project": Permission.VIEW_PROJECT,
    "get_spaces": Permission.VIEW_SPACE,
    "get_sections": Permission.VIEW_SECTION,
    "get_tasks": Permission.VIEW_TASK,
    "get_employee": Permission.VIEW_EMPLOYEE,
    "search_knowledge_base_tool": Permission.VIEW_KNOWLEDGE_BASE,
    "get_task_by_id": Permission.VIEW_TASK
}


def authorize_tool(user_id, tool_name):
    """
    Check whether the user is authorized to use a specific tool.
    """

    permission = TOOL_PERMISSIONS.get(tool_name)

    if permission is None:
        raise PermissionError(
            f"Tool '{tool_name}' is not authorized because "
            f"no permission is defined for it."
        )

    require_permission(user_id, permission)

    return True