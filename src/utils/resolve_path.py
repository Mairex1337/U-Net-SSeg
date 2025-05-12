import os


def resolve_path(path: str, levels_up: int) -> str:
    """
    Resolves a relative path to an absolute path from the project root.

    Args:
        path (str): The relative path using forward slashes ('/') as separators.
        levels_up (int): Number of parent directories to traverse up from the current file.

    Returns:
        str: The resolved absolute path.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), *['..'] * levels_up)
    )
    abs_path = os.path.join(project_root, *path.split('/'))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Resolved path does not exist: {abs_path}.",
            "Check cfg.yaml for appropriate /data dir structure."
        )
    return abs_path