"""
Prompt templates and chaining utilities.

Build reusable, composable prompt templates with variable substitution
and chain them together for multi-step LLM workflows.
"""

from __future__ import annotations

import re
from typing import Optional


class PromptTemplate:
    """A reusable prompt template with {{variable}} placeholders.

    Example:
        template = PromptTemplate(
            "Summarize this {{language}} code:\\n\\n{{code}}",
            defaults={"language": "Python"}
        )
        prompt = template.render(code="def hello(): ...")
    """

    def __init__(
        self,
        template: str,
        defaults: Optional[dict[str, str]] = None,
        name: Optional[str] = None,
    ):
        self.template = template
        self.defaults = defaults or {}
        self.name = name or "unnamed"
        self._vars = set(re.findall(r"\{\{(\w+)\}\}", template))

    @property
    def variables(self) -> set[str]:
        """Template variable names."""
        return self._vars

    @property
    def required_variables(self) -> set[str]:
        """Variables without defaults."""
        return self._vars - set(self.defaults.keys())

    def render(self, **kwargs) -> str:
        """Render the template with provided variables.

        Raises ValueError if required variables are missing.
        """
        merged = {**self.defaults, **kwargs}
        missing = self.required_variables - set(merged.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        result = self.template
        for key, value in merged.items():
            result = result.replace("{{" + key + "}}", str(value))
        return result

    def partial(self, **kwargs) -> "PromptTemplate":
        """Return a new template with some variables pre-filled."""
        new_defaults = {**self.defaults, **kwargs}
        return PromptTemplate(self.template, defaults=new_defaults, name=self.name)

    def __repr__(self) -> str:
        return f"PromptTemplate({self.name!r}, vars={self._vars})"


class PromptChain:
    """Chain multiple prompt templates together with data flow.

    Each step's output becomes available to subsequent steps.

    Example:
        chain = PromptChain()
        chain.add("research", PromptTemplate("Research {{topic}} thoroughly."))
        chain.add("summarize", PromptTemplate("Summarize: {{research}}"))

        results = chain.run(
            llm_fn=my_completion_function,
            topic="quantum computing"
        )
        print(results["summarize"])
    """

    def __init__(self):
        self._steps: list[tuple[str, PromptTemplate]] = []

    def add(self, name: str, template: PromptTemplate) -> "PromptChain":
        """Add a step to the chain."""
        self._steps.append((name, template))
        return self

    def run(self, llm_fn, **initial_vars) -> dict[str, str]:
        """Execute the chain. Each step's output is stored by its name.

        Args:
            llm_fn: A function that takes a prompt string and returns a response string.
            **initial_vars: Initial variables available to all steps.

        Returns:
            Dict mapping step names to their outputs.
        """
        context = dict(initial_vars)
        results: dict[str, str] = {}

        for name, template in self._steps:
            prompt = template.render(**context)
            response = llm_fn(prompt)
            results[name] = response
            context[name] = response  # output becomes input for next step

        return results

    @property
    def steps(self) -> list[str]:
        """Step names in order."""
        return [name for name, _ in self._steps]

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"PromptChain(steps={self.steps})"
