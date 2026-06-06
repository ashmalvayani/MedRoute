import json
import os
import threading

# Serialize all history reads/writes. With concurrent evaluation workers,
# two threads interleaving read-modify-write on the same history JSON file
# corrupt it (observed as "Extra data" / "Expecting value" JSON parse errors).
_HISTORY_LOCK = threading.Lock()


def _read_history(path):
    """Read a history JSON file. Returns [] if missing or corrupt."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A concurrent writer may have left the file half-written; don't crash.
        return []


def _atomic_write_json(path, data):
    """Write JSON via tmp+rename so concurrent readers never see a partial file."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


def memory(id, question, file_name, modality_type, answer):
    # Base dir is configurable so that different evaluation runs
    # (different LLMs, different experiments) don't clobber each other.
    history_dir = os.environ.get('MAM_HISTORY_DIR', './history')
    os.makedirs(history_dir, exist_ok=True)
    paths = {
        'text':  os.path.join(history_dir, 'text_history.json'),
        'image': os.path.join(history_dir, 'image_history.json'),
        'video': os.path.join(history_dir, 'video_history.json'),
        'audio': os.path.join(history_dir, 'audio_history.json'),
    }
    if modality_type not in paths:
        return None
    item = {"id": id, "question": question, "answer": answer}
    if modality_type != 'text':
        item["file_name"] = file_name
    path = paths[modality_type]
    # Serialize the whole read-modify-write so parallel eval workers can't
    # interleave and corrupt the JSON file.
    with _HISTORY_LOCK:
        data = _read_history(path)
        data.append(item)
        _atomic_write_json(path, data)
    return "done"
