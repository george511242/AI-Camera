from typing import Any

def truncate_long_strings(data: Any, max_length: int = 255) -> Any:
    """
    Recursively truncate string fields in data structures to max_length characters.
    
    :param data: Data structure to truncate (dict, list, or any value)
    :param max_length: Maximum length for string fields
    :return: Data structure with truncated strings
    """
    if isinstance(data, dict):
        return {key: truncate_long_strings(value, max_length) for key, value in data.items()}
    elif isinstance(data, list):
        return [truncate_long_strings(item, max_length) for item in data]
    elif isinstance(data, str) and len(data) > max_length:
        return data[:max_length] + "..."
    else:
        return data