import ast
import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


LETTERS = tuple(chr(ord("A") + i) for i in range(26))
MEDSETS_REPO_CACHE = "datasets--parthpk--Medsets"


def load_medsets_records(
    folder_names: str | Sequence[str],
    split: str,
    seed: int,
    dataset_name: str | None = None,
) -> List[Dict[str, Any]] | None:
    root = _resolve_medsets_root()
    if root is None:
        if _medsets_required():
            raise FileNotFoundError(
                "Could not find the cached parthpk/Medsets dataset. "
                "Run `hf download parthpk/Medsets --repo-type dataset` or set MEDSETS_ROOT "
                "to the downloaded Medsets directory."
            )
        return None

    try:
        dataset_dir = _find_dataset_dir(root, folder_names)
        csv_paths = _pick_csvs(dataset_dir, split, dataset_name)
    except FileNotFoundError as exc:
        if _medsets_required():
            raise FileNotFoundError(
                f"{exc}\nResolved Medsets root: {root}"
            ) from exc
        return None
    df = pd.concat((_read_medsets_csv(path) for path in csv_paths), ignore_index=True)

    split_col = _find_column(df.columns, ("split", "set", "subset"))
    if split_col:
        split_mask = df[split_col].astype(str).str.lower().eq(split.lower())
        if split_mask.any():
            df = df[split_mask].reset_index(drop=True)
    elif not any(split.lower() in path.stem.lower() for path in csv_paths):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(df))
        cut = int(0.8 * len(indices))
        picked = indices[:cut] if split == "train" else indices[cut:]
        df = df.iloc[picked].reset_index(drop=True)

    return [_row_to_record(row, dataset_dir, i, dataset_name) for i, row in df.iterrows()]


def _medsets_required() -> bool:
    return os.environ.get("MEDSETS_REQUIRED", "0").lower() in {"1", "true", "yes", "on"}


def _resolve_medsets_root() -> Path | None:
    root_env = os.environ.get("MEDSETS_ROOT")
    if root_env:
        return Path(root_env).expanduser()

    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        hub_cache = Path(cache_dir).expanduser()
    else:
        hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        hub_cache = hf_home / "hub"

    repo_cache = hub_cache / MEDSETS_REPO_CACHE
    refs_main = repo_cache / "refs" / "main"
    if refs_main.is_file():
        try:
            commit = refs_main.read_text().strip()
        except OSError:
            commit = ""
        if commit:
            snapshot = repo_cache / "snapshots" / commit
            if snapshot.is_dir():
                return snapshot

    snapshots_dir = repo_cache / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            return max(snapshots, key=lambda path: path.stat().st_mtime)

    return None


def records_to_dataframe(
    records: Iterable[Dict[str, Any]],
    sample_n: int | None,
    seed: int,
    require_mcq_options: bool = True,
) -> pd.DataFrame:
    rows = [_normalize_record(item) for item in records]
    df = pd.DataFrame(rows)
    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    _require_qa_columns(df, require_options=require_mcq_options)
    return df.reset_index(drop=True)


def format_options(record: pd.Series) -> str:
    return "".join(
        f"Option {letter}: {str(record[letter]).strip()}\n"
        for letter in option_letters(record)
    )


def option_letters(record: pd.Series) -> List[str]:
    return [
        letter for letter in LETTERS
        if letter in record.index and pd.notna(record[letter]) and str(record[letter]).strip()
    ]


def extract_choice(answer: str, valid_letters: Sequence[str] | None = None) -> str:
    letters = "".join(valid_letters or LETTERS)
    escaped = re.escape(letters)
    patterns = [
        rf"ANSWER:\s*([{escaped}])\b",
        rf"answer\s+is\s+(?:Option\s+)?(?:\**)([{escaped}])(?:\**)\b",
        rf"(?:correct|best)\s+(?:answer|option)\s*(?:is|:)\s*(?:\**)([{escaped}])(?:\**)\b",
        rf"\bOption\s+([{escaped}])\b",
        rf"\(([{escaped}])\)",
        rf"(?:^|\n)\s*\**([{escaped}])\**\s*[\.\)\:]",
        rf"(?:^|[\s,;])\**([{escaped}])\**[\.\)\:]",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    stripped = answer.strip()
    if stripped and stripped[0].upper() in letters:
        if len(stripped) == 1 or not stripped[1].isalpha():
            return stripped[0].upper()
    return stripped[:1]


def require_existing_images(df: pd.DataFrame, dataset_name: str) -> None:
    if "image_path" not in df.columns:
        raise FileNotFoundError(f"[{dataset_name}] image_path column is missing")

    paths = [str(path) for path in df["image_path"].tolist() if str(path).strip()]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        extra = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"[{dataset_name}] {len(missing)}/{len(paths)} image paths do not exist:\n"
            f"{preview}{extra}\n"
            "Set MEDSETS_ROOT to the downloaded Medsets repo root, or check that dataset zip files extracted correctly."
        )


def _require_qa_columns(df: pd.DataFrame, *, require_options: bool = True) -> None:
    if df.empty:
        raise ValueError("Medsets CSV produced no rows")

    missing = []
    if "question" not in df.columns or not df["question"].astype(str).str.strip().any():
        missing.append("question")
    if require_options and not any(
        letter in df.columns and df[letter].astype(str).str.strip().any() for letter in LETTERS
    ):
        missing.append("options")
    if "correct_answer" not in df.columns or not df["correct_answer"].astype(str).str.strip().any():
        missing.append("correct_answer")
    if "image_path" not in df.columns or not df["image_path"].astype(str).str.strip().any():
        missing.append("image_path")

    if missing:
        preview_cols = [col for col in ["question", "correct_answer", "answer_text", "image_path", *LETTERS] if col in df.columns]
        preview = df[preview_cols].head(3).to_dict(orient="records") if preview_cols else []
        raise ValueError(
            f"Medsets CSV is missing required QA fields after normalization: {', '.join(missing)}. "
            f"Normalized columns: {list(df.columns)}. Sample rows: {preview}"
        )


def _find_dataset_dir(root: Path, folder_names: str | Sequence[str]) -> Path:
    names = [folder_names] if isinstance(folder_names, str) else list(folder_names)
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate

    wanted = {_clean_name(name) for name in names}
    for child in root.iterdir():
        if child.is_dir() and _clean_name(child.name) in wanted:
            return child

    available = ", ".join(sorted(child.name for child in root.iterdir() if child.is_dir()))
    raise FileNotFoundError(
        f"Could not find Medsets folder {names} under {root}. "
        f"Available folders: {available or 'none'}"
    )


def _pick_csvs(dataset_dir: Path, split: str, dataset_name: str | None = None) -> List[Path]:
    csv_paths = sorted(dataset_dir.glob("*.csv"))
    if not csv_paths:
        csv_paths = sorted(dataset_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    split_paths = [path for path in csv_paths if split.lower() in path.stem.lower()]
    if split == "test" and not split_paths:
        split_paths = [
            path for path in csv_paths
            if "train" not in path.stem.lower() and "val" not in path.stem.lower()
        ]
    candidates = split_paths or csv_paths

    if dataset_name and dataset_name.lower() == "pmcvqa":
        filtered = [path for path in candidates if _pmcvqa_csv_has_qa_schema(path)]
        if filtered:
            return filtered
        raise FileNotFoundError(
            "PMC-VQA Medsets: no CSV looked like a QA table (question + options + answer). "
            f"This often happens when manifests or HF blob lists were matched instead of the "
            f"PMC MCQ export. Directory: {dataset_dir}. "
            f"Candidates ({len(candidates)}): {[p.name for p in candidates[:25]]}"
            + (" ..." if len(candidates) > 25 else "")
        )

    return candidates


def _pmcvqa_csv_has_qa_schema(path: Path) -> bool:
    """Reject single-column image/blob manifests that share the PMC-VQA Medsets folder."""
    try:
        peek = pd.read_csv(path, nrows=4)
    except Exception:
        return False
    if peek.empty or peek.shape[1] < 1:
        return False
    cols = list(peek.columns)

    if all(isinstance(c, int) for c in cols):
        return peek.shape[1] >= 6

    qcol = _find_column(
        cols,
        ("question", "Question", "QUESTION", "question_text", "QuestionText", "query"),
    )
    acol = _find_column(
        cols,
        (
            "correct_answer",
            "answer_label",
            "answer_idx",
            "label",
            "Answer_label",
            "gold",
            "target",
            "GoldLabel",
        ),
    )
    opt_letters = sum(1 for c in cols if _option_letter(str(c)))
    options_blob = _find_column(cols, ("options", "choices", "Choices"))
    # Single unnamed/path-only columns (common noise next to real PMC CSVs).
    if peek.shape[1] == 1 and opt_letters == 0 and options_blob is None:
        return False
    has_opts = opt_letters >= 2 or options_blob is not None
    return qcol is not None and acol is not None and has_opts


def load_medsets_pathvqa_records(split: str, seed: int) -> List[Dict[str, Any]] | None:
    """Medsets PathVQA CSVs: image path, question, short answer, caption (no header row)."""
    root = _resolve_medsets_root()
    if root is None:
        if _medsets_required():
            raise FileNotFoundError(
                "Could not find the cached parthpk/Medsets dataset. "
                "Run `hf download parthpk/Medsets --repo-type dataset` or set MEDSETS_ROOT."
            )
        return None

    try:
        dataset_dir = _find_dataset_dir(root, ("PathVQA", "pathvqa"))
        csv_paths = _pick_csvs(dataset_dir, split)
    except FileNotFoundError as exc:
        if _medsets_required():
            raise FileNotFoundError(f"{exc}\nResolved Medsets root: {root}") from exc
        return None

    df = pd.concat((_read_pathvqa_csv(path) for path in csv_paths), ignore_index=True)

    split_col = _find_column(df.columns, ("split", "set", "subset"))
    if split_col:
        split_mask = df[split_col].astype(str).str.lower().eq(split.lower())
        if split_mask.any():
            df = df[split_mask].reset_index(drop=True)
    elif not any(split.lower() in path.stem.lower() for path in csv_paths):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(df))
        cut = int(0.8 * len(indices))
        picked = indices[:cut] if split == "train" else indices[cut:]
        df = df.iloc[picked].reset_index(drop=True)

    records: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        img_cell = row["figure_path"]
        records.append(
            {
                "figure_path": _resolve_image_path(img_cell, dataset_dir),
                "question": str(row["question"]).strip(),
                "answer": str(row["answer_text"]).strip(),
                "caption": str(row["caption"]).strip() if pd.notna(row["caption"]) else "",
                "options": {},
                "question_id": i,
            }
        )
    return records


def _read_pathvqa_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"PathVQA CSV expected >= 4 columns at {path}, got {df.shape[1]}")
    out = df.iloc[:, :4].copy()
    out.columns = ["figure_path", "question", "answer_text", "caption"]
    return out


def _read_medsets_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if _looks_headerless(df.columns):
        return pd.read_csv(path, header=None)
    return df


def _looks_headerless(columns: Iterable[Any]) -> bool:
    names = [str(col).strip() for col in columns]
    if len(names) < 5:
        return False
    has_image_first = bool(re.search(r"\.(?:jpg|jpeg|png|bmp|gif|webp|dcm)$", names[0], re.IGNORECASE))
    has_answer_letter = len(names[-2]) == 1 and names[-2].upper() in LETTERS
    return has_image_first and has_answer_letter


def _row_to_record(
    row: pd.Series,
    dataset_dir: Path,
    question_id: int,
    dataset_name: str | None,
) -> Dict[str, Any]:
    if _is_headerless_row(row):
        return _headerless_row_to_record(row, dataset_dir, question_id)

    columns = row.index
    image_col = _find_column(columns, (
        "figure_path", "image_path", "image", "image_file", "image_filename",
        "filename", "file_name", "path", "image_rel", "filepath", "file_path",
        "file", "relative_path", "rel_path", "img", "img_path", "img_name",
        "image_name", "image_index", "imageid",
    )) or _infer_image_column(row)
    question_col = _find_column(columns, ("question", "Question"))
    answer_col = _find_column(columns, ("answer", "answer_text", "Answer"))
    answer_idx_col = _find_column(columns, (
        "answer_idx", "answer_label", "correct_answer", "label", "labels", "class",
        "category", "target", "answer_index", "finding", "disease", "tumor_type",
        "lesion_type",
    ))

    options = _extract_options(row)
    answer_text = str(row[answer_col]).strip() if answer_col and pd.notna(row[answer_col]) else ""
    answer_idx = _normalize_answer_idx(
        row[answer_idx_col] if answer_idx_col else "",
        options,
        answer_text,
    )
    question = str(row[question_col]).strip() if question_col and pd.notna(row[question_col]) else ""

    return {
        "figure_path": _resolve_image_path(row[image_col], dataset_dir) if image_col else "",
        "question": question,
        "answer": answer_text,
        "options": options,
        "answer_idx": answer_idx,
        "question_id": row.get("question_id", row.get("id", question_id)),
    }


def _is_headerless_row(row: pd.Series) -> bool:
    return all(isinstance(col, int) for col in row.index)


def _headerless_row_to_record(
    row: pd.Series,
    dataset_dir: Path,
    question_id: int,
) -> Dict[str, Any]:
    values = row.tolist()
    if len(values) < 6:
        raise ValueError(f"Expected at least 6 columns in headerless Medsets row, got {len(values)}")

    options = {
        LETTERS[i]: str(value).strip()
        for i, value in enumerate(values[2:-3])
        if i < len(LETTERS) and pd.notna(value) and str(value).strip()
    }

    return {
        "figure_path": _resolve_image_path(values[0], dataset_dir),
        "question": "" if pd.isna(values[1]) else str(values[1]).strip(),
        "answer": "" if pd.isna(values[-3]) else str(values[-3]).strip(),
        "options": options,
        "answer_idx": "" if pd.isna(values[-2]) else str(values[-2]).strip().upper(),
        "question_id": question_id,
    }


def _normalize_record(item: Dict[str, Any]) -> Dict[str, Any]:
    options = item.get("options", {})
    if isinstance(options, list):
        options = {LETTERS[i]: str(value) for i, value in enumerate(options[:len(LETTERS)])}
    else:
        options = {
            letter: str(v)
            for k, v in dict(options).items()
            if (letter := _option_key_to_letter(k))
        }

    answer_text = str(item.get("answer", "")).strip()
    if not options:
        answer_idx = answer_text or str(item.get("answer_idx", "")).strip()
    else:
        answer_idx = _normalize_answer_idx(item.get("answer_idx", ""), options, answer_text)
    image_path = str(item["figure_path"]).strip()
    row = {
        "question_id": item.get("question_id"),
        "question": str(item["question"]).strip(),
        "correct_answer": answer_idx,
        "answer_text": answer_text,
        "image_path": image_path,
        "image_filename": Path(image_path).name,
    }
    for letter in LETTERS:
        if letter in options and str(options[letter]).strip():
            row[letter] = str(options[letter]).strip()
    cap = item.get("caption")
    if cap is not None and str(cap).strip():
        row["caption"] = str(cap).strip()
    return row


def _extract_options(row: pd.Series) -> Dict[str, str]:
    options: Dict[str, str] = {}
    option_cols = [col for col in row.index if _option_letter(str(col))]
    for col in row.index:
        letter = _option_letter(str(col))
        if len(option_cols) >= 2 and letter and pd.notna(row[col]) and str(row[col]).strip():
            options[letter] = str(row[col]).strip()

    options_col = _find_column(row.index, ("options", "choices"))
    if options_col and pd.notna(row[options_col]):
        parsed = _parse_options_cell(row[options_col])
        options.update(parsed)

    return dict(sorted(options.items(), key=lambda item: LETTERS.index(item[0])))


def _parse_options_cell(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, list):
        return {LETTERS[i]: str(v) for i, v in enumerate(value[:len(LETTERS)])}
    else:
        text = str(value).strip()
        try:
            raw = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return {}
        if isinstance(raw, list):
            return {LETTERS[i]: str(v) for i, v in enumerate(raw[:len(LETTERS)])}
        if not isinstance(raw, dict):
            return {}

    parsed = {}
    for key, option in raw.items():
        letter = _option_key_to_letter(key)
        if letter:
            parsed[letter] = str(option)
    return parsed


def _normalize_answer_idx(value: Any, options: Dict[str, str], answer_text: str) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.upper() in options:
        return text.upper()
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(LETTERS):
            return LETTERS[idx]
    for letter, option in options.items():
        if text and text.lower() == option.lower():
            return letter
        if answer_text and answer_text.lower() == option.lower():
            return letter
    return text.upper()


def _resolve_image_path(value: Any, dataset_dir: Path) -> str:
    if value is None or pd.isna(value):
        return ""
    path = Path(str(value).strip()).expanduser()
    candidate = path if path.is_absolute() else dataset_dir / path
    if candidate.exists():
        return str(candidate.resolve())

    extracted_dir = _ensure_extracted_archives(dataset_dir)
    if extracted_dir != dataset_dir:
        extracted_candidate = extracted_dir / path
        if extracted_candidate.exists():
            return str(extracted_candidate.resolve())

        found = _find_image_by_name(str(extracted_dir), path.name)
        if found:
            return found

    found = _find_image_by_name(str(dataset_dir), path.name)
    if found:
        return found

    return str(candidate)


def _ensure_extracted_archives(dataset_dir: Path) -> Path:
    zip_paths = sorted(dataset_dir.rglob("*.zip"))
    if not zip_paths:
        return dataset_dir

    extract_root = Path(
        os.environ.get("MEDSETS_EXTRACT_DIR", "~/.cache/medroute/medsets_extracted")
    ).expanduser()
    target_dir = extract_root / dataset_dir.parent.name / dataset_dir.name
    complete_marker = target_dir / ".complete"
    if complete_marker.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        _extract_zip(zip_path, target_dir)
    complete_marker.write_text("\n".join(str(path) for path in zip_paths))
    return target_dir


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = target_dir / member.filename
            resolved = member_path.resolve()
            if target_root != resolved and target_root not in resolved.parents:
                raise ValueError(f"Unsafe path in zip {zip_path}: {member.filename}")
        archive.extractall(target_dir)


def _infer_image_column(row: pd.Series) -> str | None:
    for col in row.index:
        value = row[col]
        if pd.isna(value):
            continue
        text = str(value).strip().lower()
        if re.search(r"\.(?:jpg|jpeg|png|bmp|gif|webp|dcm)$", text):
            return str(col)
    return None


@lru_cache(maxsize=8192)
def _find_image_by_name(dataset_dir: str, filename: str) -> str:
    if not filename:
        return ""
    root = Path(dataset_dir)
    matches = sorted(root.rglob(filename))
    return str(matches[0].resolve()) if matches else ""


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    normalized = {_clean_name(str(col)): str(col) for col in columns}
    for candidate in candidates:
        found = normalized.get(_clean_name(candidate))
        if found:
            return found
    return None


def _option_letter(column: str) -> str | None:
    cleaned = _clean_name(column)
    if len(cleaned) == 1 and cleaned.upper() in LETTERS:
        return cleaned.upper()
    match = re.fullmatch(r"(?:choice|option)([a-z])", cleaned)
    if match:
        return match.group(1).upper()
    return None


def _option_key_to_letter(key: Any) -> str | None:
    text = str(key).strip()
    direct = _option_letter(text)
    if direct:
        return direct
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(LETTERS):
            return LETTERS[idx - 1]
        if idx == 0:
            return LETTERS[0]
    return None


def load_csv_as_records(path: Path, dataset_dir: Path, dataset_name: str | None) -> List[Dict[str, Any]]:
    df = _read_medsets_csv(path)
    return [_row_to_record(row, dataset_dir, i, dataset_name) for i, row in df.iterrows()]


def load_local_vision_csv_records(
    data_dir: Path,
    split: str,
    dataset_name: str,
    seed: int,
) -> List[Dict[str, Any]] | None:
    if not data_dir.is_dir():
        return None
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        return None
    split_paths = [path for path in csv_paths if split.lower() in path.stem.lower()]
    if not split_paths and split == "test":
        split_paths = [
            path for path in csv_paths
            if "train" not in path.stem.lower() and "val" not in path.stem.lower()
        ]
    if not split_paths:
        return None
    df = pd.concat((_read_medsets_csv(path) for path in split_paths), ignore_index=True)
    split_col = _find_column(df.columns, ("split", "set", "subset"))
    if split_col:
        split_mask = df[split_col].astype(str).str.lower().eq(split.lower())
        if split_mask.any():
            df = df[split_mask].reset_index(drop=True)
    elif len(split_paths) == 1 and split.lower() not in split_paths[0].stem.lower():
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(df))
        cut = int(0.8 * len(indices))
        picked = indices[:cut] if split == "train" else indices[cut:]
        df = df.iloc[picked].reset_index(drop=True)
    return [_row_to_record(row, data_dir, i, dataset_name) for i, row in df.iterrows()]


def _clean_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())
