from typing import List, Any, Dict
import re
import os

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.tools.search.wiki import search_wiki_main
from GDesigner.prompt.dynamic_prompt import generate_dynamic_prompt
import GDesigner.prompt.dynamic_prompt as dynamic_prompt_module

from pathlib import Path


def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)


# Global toggle: set via run_medqa.py --dynamic_prompts flag
DYNAMIC_PROMPTS_ENABLED = False
JUDGE_MODEL_FOR_PROMPTS = None  # set from args
PROMPT_MODEL_FOR_PROMPTS = None  # if set, use this model + BASE_URL instead of judge


@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "",
                 llm_name: str = "", dynamic_description: str = None):
        super().__init__(id, "AnalyzeAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        # Use dynamic description from panel generator if provided
        if dynamic_description:
            self.constraint = f"You are a {self.role}.\n{dynamic_description}"
            self._dynamic_description = dynamic_description
        else:
            self.constraint = self.prompt_set.get_analyze_constraint(self.role)
            self._dynamic_description = None
        self.wiki_summary = ""

    async def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict],
                              temporal_info: Dict[str, Dict], **kwargs) -> List[Any]:
        # Dynamic prompt: replace generic constraint with question-specific guidance
        if DYNAMIC_PROMPTS_ENABLED and JUDGE_MODEL_FOR_PROMPTS:
            # Use prompt_model (e.g. 8B on BASE_URL) if set, otherwise judge model
            if PROMPT_MODEL_FOR_PROMPTS:
                prompt_model = PROMPT_MODEL_FOR_PROMPTS
                dynamic_prompt_module.PROMPT_BASE_URL = os.getenv('BASE_URL', 'http://localhost:8000')
            else:
                prompt_model = JUDGE_MODEL_FOR_PROMPTS
                dynamic_prompt_module.PROMPT_BASE_URL = None  # use JUDGE_BASE_URL
            # For dynamic pool, use the panel-generated description; otherwise use static
            if self._dynamic_description:
                base_desc = self._dynamic_description
            else:
                base_desc = self.prompt_set.get_description(self.role) if hasattr(self.prompt_set, 'get_description') else ""
            # When VLM sees the image, use task_plain (no caption needed)
            img_path = raw_inputs.get('image_path')
            prompt_question = raw_inputs.get('task_plain', raw_inputs['task']) if img_path else raw_inputs['task']
            dynamic_constraint = await generate_dynamic_prompt(
                judge_model=prompt_model,
                question=prompt_question,
                specialist_role=self.role,
                base_description=base_desc or self.constraint,
                image_path=img_path,
            )
            # Append the standard MCQ instructions after the dynamic guidance
            system_prompt = dynamic_constraint + "\n" + self.prompt_set.get_constraint()
        else:
            system_prompt = f"{self.constraint}"

        # Store for logging in graph.py
        self._last_system_prompt = system_prompt

        user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role != 'Fake' else self.prompt_set.get_adversarial_answer_prompt(raw_inputs['task'])
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            if self.role == 'Wiki Searcher' and info['role'] == 'Knowlegable Expert':
                queries = find_strings_between_pluses(info['output'])
                wiki = await search_wiki_main(queries)
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    user_prompt += f"The key entities of the problem are explained in Wikipedia as follows:{self.wiki_summary}"
            spatial_str += f"{info['role']}: {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"{info['role']}: {info['output']}\n\n"

        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(temporal_str) else ""
        return system_prompt, user_prompt

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict],
                             temporal_info: Dict[str, Dict], **kwargs):
        def _make_user_content(user_text: str, image_path: str | None):
            if not image_path:
                return user_text
            # Use file:// URL so vLLM reads the image directly (avoids base64 context bloat)
            abs_path = str(Path(image_path).resolve())
            return [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
            ]

        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)

        image_path = input.get("image_path", None)
        user_prompt = _make_user_content(user_prompt, image_path)
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.llm.agen(message)
        response = response.split('<|end_of_turn|>')[0].strip()
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""
        print(f"################system prompt:{system_prompt}")
        print(f"################user prompt:{user_prompt}")
        print(f"################response:{response}")
        return response
