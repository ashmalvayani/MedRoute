import glob
import re
import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC
import os

class PubMedQADataset(ABC):
    def __init__(self,
        split: str,
        ) -> None:
        # import ipdb; ipdb.set_trace()
        self._split = split

        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PubMedQA', 'data', self._split) + "/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'pubmedqa'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:
    
        rng = np.random.default_rng(888)
    
        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        
        print("Number of topics: ", len(csv_paths))
    
        names = ['question', 'A', 'B', 'C', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            probe = pd.read_csv(path, header=None, nrows=1, encoding='utf-8')
            if probe.shape[1] >= 6:
                col_names = names + ['context']
            else:
                col_names = names
            single_df = pd.read_csv(path, header=None,
                            names=col_names, encoding='utf-8')
            total_df = pd.concat([total_df, single_df])
    
        total_df = total_df.reset_index(drop=True)
    
        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))
    
        print("Total number of questions: ", len(total_df))
    
        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        context = record.get('context', None)
        if pd.notna(context) and context:
            demo_question = (
                f"Context:\n{context}\n\n"
                f"Question: {record['question']}\n"
                f"Option A: {record['A']}\n"
                f"Option B: {record['B']}\n"
                f"Option C: {record['C']}\n"
            )
        else:
            demo_question = (
                f"{record['question']}\n"
                f"Option A: {record['A']}\n"
                f"Option B: {record['B']}\n"
                f"Option C: {record['C']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                answer = answer[ans_pos+len("answer is"):].strip(":").strip().strip("Option").strip()
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer

    @staticmethod
    def record_to_target_check(ground_truth_answer, predicted_answer, question) -> int:
        """Simple regex-based letter matching for MCQ (A-C). No GPT required."""
        predicted_answer = predicted_answer.replace('assistant', 'Option').strip()
        gt = ground_truth_answer.strip().upper()

        patterns = [
            r'answer\s+is\s+(?:Option\s+)?([A-C])',
            r'\bOption\s+([A-C])\b',
            r'\(([A-C])\)',
            r'\b([A-C])\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, predicted_answer, re.IGNORECASE)
            if match:
                return 1 if match.group(1).upper() == gt else 0

        if predicted_answer and predicted_answer[0].upper() in 'ABC':
            return 1 if predicted_answer[0].upper() == gt else 0

        return 0

    @staticmethod
    def _record_to_target_check_gpt_unused(ground_truth_answer, predicted_answer, question) -> str:
        # Kept for reference only - requires Azure OpenAI
        system_prompt = (
        "You are a strict and careful evaluator for medical multiple-choice question answering tasks."
        )
        user_prompt = f"""Task:
Determine whether the model's predicted answer matches the ground-truth answer.

Inputs:
Question:
{question}

Ground-truth answer:
{ground_truth_answer}

Predicted answer:
{predicted_answer}

Evaluation guidelines:
1) Extract the final chosen option from the predicted answer.
   - Prefer an explicit option index/label if stated (e.g., "Option 2", "2", "(B)", "B").
   - If multiple options are mentioned, use the final decision the model commits to (e.g., "Therefore, ...", "Final answer: ...").
2) If no explicit option is stated, infer the implied choice from the prediction's conclusion.
   - Ignore restating the question/options or tentative analysis unless it clearly commits.
3) Compare the extracted/implied choice to the ground-truth choice.
   - Mark "correct" only if they refer to the same option.
   - Otherwise mark "incorrect".
4) Be conservative: if the predicted answer is ambiguous or does not commit to a single option, mark "incorrect".

Output format (STRICT):
Return ONLY a valid JSON object with exactly these keys:
- "result": "correct" or "incorrect"
- "reason": a brief explanation (one sentence)

Do not output any other text.
""" 
        
    #     user_prompt = f"""
    # Compare the ground truth and predicted answers below.
    
    # Question:
    # "{question}"
    
    # Ground truth answer:
    # "{ground_truth_answer}"
    
    # Predicted answer:
    # "{predicted_answer}"
    
    # Evaluation rules:
    # - The predicted answer will have the final chosen option index and explanation in the answer text.
    # - If the predicted answer clearly states an option, compare it directly.
    # - Otherwise, compare meaning conservatively.
    # - Respond only in strict JSON, if the answer is 'correct' or 'incorrect'.
    # - Give a short reason for your answer as well.
    # """
    
        try:
            completion = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=256,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "EvalResult",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "result": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["result", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
    
            result_json = json.loads(completion.choices[0].message.content)
            print(result_json)
            if result_json['result'].lower() == 'correct':
                return 1
            return 0
        except Exception as e:
            # raise ValueError(f"Error: {e}")
            print(f"Error: {e}")
            print(f"GPT Response: {completion}")
            return 0
            
            
