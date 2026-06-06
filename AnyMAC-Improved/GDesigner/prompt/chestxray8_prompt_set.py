from GDesigner.prompt.pmcvqa_prompt_set import PMCVQAPromptSet, SPECIALISTS
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


@PromptSetRegistry.register("chestxray8")
class ChestXray8PromptSet(PMCVQAPromptSet):
    pass
