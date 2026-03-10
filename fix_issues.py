"""Script to automatically fix common issues."""

import re
from pathlib import Path


def add_missing_imports(file_path):
    """Add common missing imports."""
    with open(file_path, "r") as f:
        content = f.read()

    # Check for missing typing imports
    if "typing." not in content and any(
        x in content for x in ["List", "Dict", "Optional", "Tuple", "Union", "Callable"]
    ):
        content = (
            "from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union\n" + content
        )

    # Check for missing numpy
    if "np." in content and "import numpy" not in content:
        content = "import numpy as np\n" + content

    # Check for missing re
    if "re." in content and "import re" not in content:
        content = "import re\n" + content

    # Check for missing time
    if "time." in content and "import time" not in content:
        content = "import time\n" + content

    with open(file_path, "w") as f:
        f.write(content)


def remove_trailing_whitespace(file_path):
    """Remove trailing whitespace from each line."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    lines = [line.rstrip() + "\n" for line in lines]

    with open(file_path, "w") as f:
        f.writelines(lines)


# Process all Python files
for py_file in Path(".").rglob("*.py"):
    print(f"Processing {py_file}")
    try:
        add_missing_imports(py_file)
        remove_trailing_whitespace(py_file)
    except Exception as e:
        print(f"Error processing {py_file}: {e}")
