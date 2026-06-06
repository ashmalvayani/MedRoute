"""
PMC-VQA prompt set for medical visual question answering.

Reuses the same medical specialist pool as MedQA/PubMedQA but with:
- 4-option MCQ (A-D) constraints
- Vision-aware prompts that reference image descriptions
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Tuple, Union

from GDesigner.prompt.prompt_set import PromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

# Reuse the same specialist pool and descriptions from pubmedqa
from GDesigner.prompt.pubmedqa_prompt_set import (
    SPECIALISTS,
    ROLE_DESCRIPTION,
)

# Toggle: let DM reason before answering (chain-of-thought)
DM_COT = False

ROLE_CONNECTION: List[Tuple[str, str]] = [
    (a, b) for a in SPECIALISTS for b in SPECIALISTS if a != b
]

ROLES: Iterable[str] = itertools.cycle(SPECIALISTS)


@PromptSetRegistry.register("pmcvqa")
class PMCVQAPromptSet(PromptSet):
    """PromptSet for PMC-VQA: medical visual QA with 4 options (A-D).

    Agents receive an image description (caption) + question + 4 options.
    The constraint and format are adapted for 4-choice MCQ.
    """

    def get_role(self) -> str:
        return next(ROLES)  # type: ignore[arg-type]

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."

    @staticmethod
    def get_constraint():
        return """
            I will give you a description of a medical image and ask you a question about it.
            I will also give you 4 answers enumerated as A, B, C, and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Your response must be one of the 4 letters: A, B, C, or D
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C, or D)
        """

    def get_role_connection(self):
        return ROLE_CONNECTION

    def get_description(self, role):
        return ROLE_DESCRIPTION.get(role, f"You are a {role}.")

    @staticmethod
    def get_analyze_constraint(role):
        return ROLE_DESCRIPTION.get(role, f"You are a {role}.") + """
I will give you a description of a medical image and ask you a question about it.
I will also give you 4 answers enumerated as A, B, C, and D.
Only one answer out of the offered 4 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C, or D)
"""

    @staticmethod
    def get_decision_constraint():
        if DM_COT:
            return """
        You are looking at a medical image. You will receive a question with 4 answer options (A, B, C, D) and opinions from specialist agents.

        Your job:
        1. Look at the image carefully — identify the modality, key findings, and abnormalities
        2. Consider each specialist's reasoning — note agreements and disagreements
        3. Use your own visual assessment to resolve conflicts between specialists
        4. Provide brief reasoning (under 100 words) explaining your visual findings and why you chose your answer
        5. End your response with exactly: "ANSWER: X" where X is A, B, C, or D

        You MUST end with "ANSWER: X" on the last line.
        """
        return """
        I will give you a description of a medical image and ask you a question about it.
        I will also give you 4 answers enumerated as A, B, C, and D.
        Only one answer out of the offered 4 is correct.
        You must choose the correct answer to the question.
        Your response must be one of the 4 letters: A, B, C, or D,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """

    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example: A, B, C, or D)
                """

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        raise NotImplementedError

    @staticmethod
    def get_decision_few_shot():
        return ""

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if not answer:
            return answer
        if DM_COT:
            # Extract from "ANSWER: X" pattern
            import re
            m = re.search(r'ANSWER:\s*([A-D])', answer, re.IGNORECASE)
            if m:
                return m.group(1).upper()
            # Fallback: last single letter A-D on its own line
            for line in reversed(answer.strip().splitlines()):
                line = line.strip()
                if len(line) == 1 and line.upper() in "ABCD":
                    return line.upper()
        # Default: first character
        return answer[0]
