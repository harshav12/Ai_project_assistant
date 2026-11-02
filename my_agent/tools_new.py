import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

TASKS_FILE = DATA_DIR.parent / "Data_processed" / "tasks.json"
WORKSPACES_FILE = DATA_DIR.parent / "Data_processed" / "workspaces.json"
SPACES_FILE = DATA_DIR.parent / "Data_processed" / "spaces.json"
SECTIONS_FILE = DATA_DIR.parent / "Data_processed" / "sections.json"
RESOURCES_FILE = DATA_DIR.parent / "Data_processed" / "resources.json"


with open(TASKS_FILE, "r", encoding="utf-8") as file:
    tasks = json.load(file)

with open(WORKSPACES_FILE, "r", encoding="utf-8") as file:
    workspaces = json.load(file)

with open(SPACES_FILE, "r", encoding="utf-8") as file:
    spaces = json.load(file)

with open(SECTIONS_FILE, "r", encoding="utf-8") as file:
    sections = json.load(file)

with open(RESOURCES_FILE, "r", encoding="utf-8") as file:
    resources = json.load(file)

sections_by_id = {
    str(section.get("section_id", "")).casefold(): section
    for section in sections
}
spaces_by_id = {
    str(space.get("space_id", "")).casefold(): space
    for space in spaces
}
workspace_ids = {
    str(workspace.get("workspace_id", "")).casefold()
    for workspace in workspaces
}


# get projects
def get_projects(status=None):
    """Return processed workspaces, treating each workspace as a project."""

    if status is None:
        return workspaces

    requested_status = status.strip().lower()

    return [
        workspace
        for workspace in workspaces
        if str(workspace.get("status", "")).lower() == requested_status
    ]



# get project


def get_project(
    project_id: str | None = None,
    project_name: str | None = None
):
    """Look up a project by Workspace ID or workspace name.

    Rules:
    - project_id accepts the Workspace ID; records retain the workspace_id key.
    - project_name accepts an exact or partial workspace name.
    - Partial names resolve only when they identify one workspace.
    - If both are provided, project_id is checked first, then project_name.
    """
    if project_id is None and project_name is None:
        raise ValueError("Provide project_id or project_name.")

    raw_project_id = str(project_id).strip() if project_id is not None else None
    raw_project_name = str(project_name).strip() if project_name is not None else None

    # A project_id argument is the workspace_id in the processed data.
    if raw_project_id:
        normalized_id = raw_project_id.casefold()
        for workspace in workspaces:
            if str(workspace.get("workspace_id", "")).casefold() == normalized_id:
                return workspace

    # Prefer an exact workspace-name match.
    if raw_project_name:
        normalized_name = raw_project_name.casefold()
        exact_matches = [
            workspace
            for workspace in workspaces
            if str(workspace.get("name", "")).strip().casefold()
            == normalized_name
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return None

    # Allow a partial name only when it is unambiguous.
    if raw_project_name:
        partial_matches = [
            workspace
            for workspace in workspaces
            if normalized_name
            in str(workspace.get("name", "")).strip().casefold()
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]

    return None


def get_spaces(workspace_id=None, space_id=None, space_name=None, status=None):
    """List Spaces with optional Workspace, ID, name, and status filters."""

    filtered_spaces = spaces

    if workspace_id is not None:
        normalized_workspace_id = str(workspace_id).strip().casefold()
        if normalized_workspace_id not in workspace_ids:
            return []
        filtered_spaces = [
            space
            for space in filtered_spaces
            if str(space.get("workspace_id") or "").casefold()
            == normalized_workspace_id
        ]

    if space_id is not None:
        normalized_space_id = str(space_id).strip().casefold()
        filtered_spaces = [
            space
            for space in filtered_spaces
            if str(space.get("space_id") or "").casefold()
            == normalized_space_id
        ]

    if space_name is not None:
        normalized_space_name = str(space_name).strip().casefold()
        filtered_spaces = [
            space
            for space in filtered_spaces
            if normalized_space_name
            in str(space.get("name") or "").strip().casefold()
        ]

    if status is not None:
        normalized_status = str(status).strip().casefold()
        filtered_spaces = [
            space
            for space in filtered_spaces
            if str(space.get("status") or "").strip().casefold()
            == normalized_status
        ]

    return filtered_spaces


def get_sections(
    space_id=None,
    section_id=None,
    section_name=None,
    workspace_id=None,
    status=None
):
    """List Sections with optional Space, Workspace, ID, name, and status filters.

    Section records retain their Workspace/Space consistency fields. When a
    Workspace filter is supplied, only Sections with consistent hierarchy
    links through a known Space are included.
    """

    filtered_sections = sections

    if workspace_id is not None:
        normalized_workspace_id = str(workspace_id).strip().casefold()
        if normalized_workspace_id not in workspace_ids:
            return []
        filtered_sections = [
            section
            for section in filtered_sections
            if _section_belongs_to_workspace(section, normalized_workspace_id)
        ]

    if space_id is not None:
        normalized_space_id = str(space_id).strip().casefold()
        if normalized_space_id not in spaces_by_id:
            return []
        filtered_sections = [
            section
            for section in filtered_sections
            if str(section.get("space_id") or "").casefold()
            == normalized_space_id
        ]

    if section_id is not None:
        normalized_section_id = str(section_id).strip().casefold()
        filtered_sections = [
            section
            for section in filtered_sections
            if str(section.get("section_id") or "").casefold()
            == normalized_section_id
        ]

    if section_name is not None:
        normalized_section_name = str(section_name).strip().casefold()
        filtered_sections = [
            section
            for section in filtered_sections
            if normalized_section_name
            in str(section.get("name") or "").strip().casefold()
        ]

    if status is not None:
        normalized_status = str(status).strip().casefold()
        filtered_sections = [
            section
            for section in filtered_sections
            if str(section.get("status") or "").strip().casefold()
            == normalized_status
        ]
        

    return filtered_sections


def _section_belongs_to_workspace(section, workspace_id):
    space_id = str(section.get("space_id") or "").casefold()
    space = spaces_by_id.get(space_id)
    if space is None:
        return False

    return (
        str(section.get("workspace_id") or "").casefold() == workspace_id
        and str(space.get("workspace_id") or "").casefold() == workspace_id
        and section.get("space_workspace_id") == section.get("workspace_id")
        and section.get("workspace_link_consistent") is True
    )


# get tasks
def get_tasks(project_id=None, status=None, task_name = None):
    """Return tasks, optionally filtered by Workspace ID and task status and task_name. 

    Task records have no Workspace ID. Workspace membership is resolved through
    SectionID, and only consistent Section/Space relationships are included
    when filtering by Workspace.

    Task-name matching is case-insensitive and supports partial matches.
    """

    filtered_tasks = tasks

    if project_id is not None:
        normalized_id = str(project_id).strip().casefold()
        if not normalized_id:
            raise ValueError("project_id must be a Workspace ID.")
        if normalized_id not in workspace_ids:
            return []

        filtered_tasks = [
            task
            for task in filtered_tasks
            if _task_belongs_to_workspace(task, normalized_id)
        ]

    if status is not None:
        normalized_status = _normalize_task_status(status)

        filtered_tasks = [
            task
            for task in filtered_tasks
            if _normalize_task_status(task.get("Status", "")) == normalized_status
        ]

    if task_name is not None:
        normalized_name = str(task_name).strip.casefold()

        if not normalized_name:
            return []

        filtered_tasks = [
            task
            for task in filtered_tasks
            if normalized_name
            in str(task.get("TaskName") or "").strip().casefold()
        ]

    return filtered_tasks


def _normalize_task_status(status):
    return "_".join(str(status).strip().casefold().split())


def _task_belongs_to_workspace(task, workspace_id):
    section = sections_by_id.get(
        str(task.get("SectionID") or "").casefold()
    )
    if section is None:
        return False

    section_workspace_id = str(section.get("workspace_id") or "").casefold()
    section_space_id = section.get("space_id")
    return (
        section_workspace_id == workspace_id
        and section_workspace_id in workspace_ids
        and section.get("workspace_link_consistent") is True
        and section.get("space_workspace_id") == section.get("workspace_id")
        and task.get("SpaceID") == section_space_id
        and task.get("section_space_id") == section_space_id
    )



# get employee
def get_employee(
    employee_id=None,
    employee_name=None,
    department=None
):
    """Retrieve PerXL resources using optional ID, name, or department filters."""

    filtered_resources = resources

    if employee_id is not None:
        normalized_id = str(employee_id).strip().upper()

        filtered_resources = [
            resource
            for resource in filtered_resources
            if str(resource.get("ResourceID", "")).strip().upper()
            == normalized_id
        ]

    if employee_name is not None:
        normalized_name = str(employee_name).strip().lower()

        filtered_resources = [
            resource
            for resource in filtered_resources
            if normalized_name
            in f"{resource.get('Name', '')} {resource.get('LastName', '')}".strip().lower()
        ]

    if department is not None:
        normalized_department = str(department).strip().lower()

        filtered_resources = [
            resource
            for resource in filtered_resources
            if str(resource.get("Department") or "").strip().lower()
            == normalized_department
        ]

    if employee_id is not None:
        return filtered_resources[0] if filtered_resources else None

    if employee_name is not None and len(filtered_resources) == 1:
        return filtered_resources[0]

    return filtered_resources


# get task by the id

def get_task_by_id(task_id):
    """Return one task by its processed task ID, or None if not found."""
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id is required")

    normalized_id = task_id.strip().casefold()

    return next(
        (
            task for task in tasks
            if str(task.get("_id", "")).strip().casefold() == normalized_id
        ),
        None
    )