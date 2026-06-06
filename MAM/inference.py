import argparse
import json
import os

from pipeline.modality_selection import modality_selection
from pipeline.type_classification import type_classification
from pipeline.role_generation import generate_role
from pipeline.web_search_check import WebSearchCheck
from pipeline.meeting import roles_meeting
from pipeline.diagnosis import final_diagnosis
from pipeline.review import review_all
from pipeline.memory import memory


def _history_dir():
    return os.environ.get('MAM_HISTORY_DIR', './history')


def _history_files():
    d = _history_dir()
    return {
        'text':  os.path.join(d, 'text_history.json'),
        'image': os.path.join(d, 'image_history.json'),
        'video': os.path.join(d, 'video_history.json'),
        'audio': os.path.join(d, 'audio_history.json'),
    }


def ensure_history_dir():
    os.makedirs(_history_dir(), exist_ok=True)


def load_history(modality_type):
    """Load relevant history records as context string.

    Tolerant of corrupt / partially-written history files — concurrent
    evaluation workers can race on the write side; if we ever observe a
    malformed JSON file we simply treat history as empty rather than crash
    the whole pipeline for that sample.
    """
    history_file = _history_files().get(modality_type)
    if not history_file or not os.path.exists(history_file):
        return ''
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ''
    if not data:
        return ''
    # Return last 3 records as reference context
    recent = data[-3:]
    items = []
    for item in recent:
        items.append(f"Q: {item.get('question', '')} A: {item.get('answer', '')}")
    return '\n'.join(items)


def parse_type_result(type_result):
    """Normalize type_classification() return value to a single string."""
    if isinstance(type_result, tuple):
        # image returns (modality_type, body_part), e.g. ('CT', 'Lung')
        return ', '.join(str(t) for t in type_result)
    return str(type_result)


def run_pipeline(step_id, question, file_name, sample_id=None):
    """Run the MAM pipeline and return a full trace dict.

    Args:
        step_id:   pipeline step to run up to (1-9).
        question:  user question / prompt.
        file_name: path to image/audio/video, or "" for text.
        sample_id: stable identifier for this sample (e.g. the MedQA index).
                   Stored with the history record in step 9 so entries can be
                   traced back to the source sample. Falls back to `step_id`
                   (upstream behavior) when not provided.

    Returns a dict with these top-level keys:
        step_id, question, file_name,
        step_1_modality_selection, step_2_type_classification,
        step_3_role_generation,    step_4_web_search,
        step_5_load_history,       step_6_multi_agent_meeting,
        step_7_final_diagnosis,    step_8_review,
        step_9_memory,
        modality_type, type_name, diagnosis, review_result, meeting
            (convenience top-level fields — kept for callers that already
             consume the flat view, e.g. extract_letter / compact predictions.)
    """
    ensure_history_dir()

    trace = {
        "step_id": step_id,
        "question": question,
        "file_name": file_name,
    }

    modality_type = None
    type_name = None
    roles_generated = None
    search_result = None
    history_item = ''
    meeting = None
    diagnosis = None
    review_result = None

    print(f"\n========== MAM Inference Pipeline ==========")
    print(f"step_id   : {step_id}")
    print(f"question  : {question}")
    print(f"file_name : {file_name}")
    print(f"============================================\n")

    # ------------------------------------------------------------------
    # Step 1: Modality Selection
    # ------------------------------------------------------------------
    if step_id >= 1:
        print("[Step 1] Modality Selection...")
        modality_type = modality_selection(question, file_name)
        print(f"  => modality_type: {modality_type}")
        trace["step_1_modality_selection"] = {
            "modality_type": modality_type,
        }

    # ------------------------------------------------------------------
    # Step 2: Type Classification
    # ------------------------------------------------------------------
    if step_id >= 2:
        print("[Step 2] Type Classification...")
        type_result = type_classification(modality_type, question, file_name)
        type_name = parse_type_result(type_result)
        print(f"  => type_name: {type_name}")
        trace["step_2_type_classification"] = {
            "type_name": type_name,
            "raw": type_result if not isinstance(type_result, tuple) else list(type_result),
        }

    # ------------------------------------------------------------------
    # Step 3: Role Generation
    # ------------------------------------------------------------------
    if step_id >= 3:
        print("[Step 3] Role Generation...")
        roles_generated = generate_role(type_name, modality_type, question, file_name)
        print(f"  => roles_generated:\n{roles_generated}")
        trace["step_3_role_generation"] = {
            "roles_generated": roles_generated,
        }

    # ------------------------------------------------------------------
    # Step 4: Web Search
    # ------------------------------------------------------------------
    if step_id >= 4:
        print("[Step 4] Web Search Check...")
        step4 = {"search_result": None, "error": None}
        try:
            search_result = WebSearchCheck(question, file_name, modality_type)
            step4["search_result"] = search_result
            print(f"  => search_result:\n{search_result}")
        except Exception as e:
            print(f"  [Warning] Web search failed: {e}. Skipping.")
            search_result = ''
            step4["error"] = str(e)
        trace["step_4_web_search"] = step4

    # ------------------------------------------------------------------
    # Step 5: Load History
    # ------------------------------------------------------------------
    if step_id >= 5:
        print("[Step 5] Loading History...")
        history_item = load_history(modality_type)
        if search_result:
            history_item = f"Web search reference:\n{search_result}\n\nHistory reference:\n{history_item}"
        print(f"  => history_item (truncated): {history_item[:200]}")
        trace["step_5_load_history"] = {
            "history_item": history_item,
        }

    # ------------------------------------------------------------------
    # Step 6: Multi-Agent Meeting
    # ------------------------------------------------------------------
    if step_id >= 6:
        print("[Step 6] Multi-Agent Meeting...")
        meeting = roles_meeting(
            question, file_name, modality_type, type_name,
            roles_generated, history_item
        )
        print(f"\n  => meeting.text (truncated): {meeting['text'][:300]}")
        print(f"  => meeting.verdict: {meeting.get('verdict')}  rounds: {len(meeting.get('rounds', []))}")
        trace["step_6_multi_agent_meeting"] = {
            "parsed_roles": meeting.get("parsed_roles"),
            "verdict": meeting.get("verdict"),
            "rounds": meeting.get("rounds"),
            "text": meeting.get("text"),
        }

    # ------------------------------------------------------------------
    # Step 7: Final Diagnosis
    # ------------------------------------------------------------------
    if step_id >= 7:
        print("[Step 7] Final Diagnosis...")
        meeting_record_text = meeting["text"] if isinstance(meeting, dict) else meeting
        diagnosis = final_diagnosis(
            question, file_name, modality_type, type_name, meeting_record_text
        )
        print(f"  => diagnosis: {diagnosis}")
        trace["step_7_final_diagnosis"] = {
            "diagnosis": diagnosis,
        }

    # ------------------------------------------------------------------
    # Step 8: Review
    # ------------------------------------------------------------------
    if step_id >= 8:
        print("[Step 8] Review...")
        review_result = review_all(
            question, file_name, modality_type, type_name, diagnosis
        )
        print(f"  => review_result: {review_result}")
        trace["step_8_review"] = {
            "review_result": review_result,
        }

    # ------------------------------------------------------------------
    # Step 9: Save to Memory
    # ------------------------------------------------------------------
    if step_id >= 9:
        print("[Step 9] Saving to Memory...")
        record_id = sample_id if sample_id is not None else step_id
        memory(record_id, question, file_name, modality_type, diagnosis)
        saved_to = _history_files().get(modality_type)
        print(f"  => Saved to {saved_to}")
        trace["step_9_memory"] = {
            "saved_to": saved_to,
            "record_id": record_id,
        }

    print("\n========== Pipeline Complete ==========")
    print(f"Final Diagnosis:\n{diagnosis}")
    print(f"Review Result  : {review_result}")
    print("=======================================\n")

    # Convenience top-level fields for downstream consumers.
    trace["modality_type"] = modality_type
    trace["type_name"] = type_name
    trace["meeting"] = meeting
    trace["diagnosis"] = diagnosis
    trace["review_result"] = review_result
    return trace


def main():
    parser = argparse.ArgumentParser(
        description='MAM: Modular Multi-Agent Framework Inference'
    )
    parser.add_argument('--step_id', type=int, default=9,
                        help='Run pipeline up to this step (1-9, default: 9 = full pipeline)')
    parser.add_argument('--question', type=str, required=True,
                        help='Medical question to answer')
    parser.add_argument('--file_name', type=str, default='',
                        help='Path to input file (image/audio/video); leave empty for text-only')
    args = parser.parse_args()

    run_pipeline(args.step_id, args.question, args.file_name)


if __name__ == '__main__':
    main()
