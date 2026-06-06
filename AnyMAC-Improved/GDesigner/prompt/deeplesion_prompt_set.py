from GDesigner.prompt.pmcvqa_prompt_set import PMCVQAPromptSet, ROLE_CONNECTION, ROLE_DESCRIPTION, ROLES, SPECIALISTS
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


@PromptSetRegistry.register("deeplesion")
class DeepLesionPromptSet(PMCVQAPromptSet):
    pass
