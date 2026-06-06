from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple, Union

from GDesigner.prompt.pmcvqa_prompt_set import PMCVQAPromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

from GDesigner.prompt.pubmedqa_prompt_set import ROLE_DESCRIPTION, SPECIALISTS

from GDesigner.prompt import pmcvqa_prompt_set as pmcvqa_ps

__all__ = ["PathVQAPromptSet", "ROLES", "SPECIALISTS"]

ROLE_CONNECTION: List[Tuple[str, str]] = [
    (a, b) for a, b in itertools.product(SPECIALISTS, SPECIALISTS) if a != b
]
ROLES: Iterable[str] = itertools.cycle(SPECIALISTS)


@PromptSetRegistry.register("pathvqa")
class PathVQAPromptSet(PMCVQAPromptSet):
    """PathVQA is short open QA (varied answer types), not 4-choice MCQ — do not inherit MCQ letter prompts."""

    def get_role_connection(self):
        return ROLE_CONNECTION

    @staticmethod
    def get_constraint():
        return """
            You are answering questions about a pathology image.
            Respond briefly and directly using only what is visible in the image.
            Do not list multiple hypotheses unless the question asks for them.
            Put the minimal correct answer on the first line (whatever form fits the question).
            You may add one short line of justification after that. Keep the total under 100 words.
        """

    @staticmethod
    def get_analyze_constraint(role):
        base = ROLE_DESCRIPTION.get(role, f"You are a {role}.")
        return (
            base
            + """
You will see a pathology image and a question. This is not multiple-choice: do not answer with only A, B, C, or D.
Give your best brief answer from the visual evidence. Other agents may share opinions — use them with critical thinking.
Put your final answer on the first line in whatever form the question expects (word, phrase, label, count, etc.). Keep under 100 words.
"""
        )

    @staticmethod
    def get_decision_constraint():
        if pmcvqa_ps.DM_COT:
            return """
        You are deciding the answer to a pathology image question (open answer unless the question lists explicit options).

        Your job:
        1. Use the image — identify the main findings relevant to the question
        2. Compare specialist opinions and note disagreements
        3. Give brief reasoning (under 80 words)
        4. End with exactly: "ANSWER: ..." with the minimal correct wording for the question

        You MUST end with a line of the form ANSWER: ...
        """
        return """
        You synthesize specialist opinions about a pathology image question.
        There are no A/B/C/D options unless the question explicitly lists them — output only the final short answer the question asks for.
        Put only that answer on the first line. Do not reply with a lone multiple-choice letter unless options are given.
        """

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a deliberately wrong answer and misleading reasoning for this pathology question: {question}
You may see other agents' outputs — still mislead. Under 100 words.
Put the wrong short answer on the first line (not a multiple-choice letter unless the question lists options)."""

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if not answer:
            return answer
        if pmcvqa_ps.DM_COT:
            import re

            m = re.search(r"ANSWER:\s*(.+)$", answer, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
        return answer.strip()
