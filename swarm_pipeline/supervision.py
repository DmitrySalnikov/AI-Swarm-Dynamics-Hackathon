"""File I/O for supervisors, separate from the frozen experiment runner."""
import json
import os
from pathlib import Path
import tempfile
import time


def read_json(path):
    # Close the OS handle before parsing; status writers still retry read races.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def progress_from_calls(directory):
    """Observe immutable journals instead of opening the worker's checkpoint."""
    directory = Path(directory)
    events = sorted((directory / "calls").glob("*/event.json"))
    requests = sorted((directory / "calls").glob("*/request.json"))
    if not requests:
        return {"calls": 0, "phase": None, "pending": None}
    latest = read_json(requests[-1])
    completed = len(events)
    return {"calls": completed, "phase": latest["context"]["phase"],
            "pending": latest["call"] if latest["call"] >= completed else None}


def write_json(path, value, attempts=20):
    """Use a unique temporary file and bounded retries for Windows lock races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.1)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                pass
