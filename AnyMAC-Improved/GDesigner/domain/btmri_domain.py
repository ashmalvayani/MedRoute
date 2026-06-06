from dataclasses import dataclass

from GDesigner.domain.pmcvqa_domain import extract_mcq_letter


@dataclass
class BtmriDomain:
    name: str = "btmri"

    def build_task(self, input_dict):
        return str(input_dict.get("task", "")).strip()

    def postprocess_answer(self, text: str) -> str:
        return extract_mcq_letter(text) or ""

    def get_final_answer(self, model_output: str) -> str:
        return self.postprocess_answer(model_output)
