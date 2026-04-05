"""
examples/prompt_chain.py — Multi-step prompt chaining demo.

Demonstrates:
  - PromptTemplate variable injection
  - PromptChain sequential execution
  - Passing previous output into next step
  - Real OpenAI calls (requires OPENAI_API_KEY)

Run:
    python examples/prompt_chain.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from llm_toolkit.prompts import PromptTemplate, PromptChain, build_messages

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def simple_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Thin wrapper around OpenAI chat completions."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ── Step definitions ──────────────────────────────────────────────────────────

step1_tpl = PromptTemplate(
    "You are a creative writing assistant. Write a 2-sentence premise for a "
    "short story about: {topic}. Be specific and original."
)

step2_tpl = PromptTemplate(
    "Expand the following story premise into a 3-paragraph outline with a "
    "clear beginning, middle, and end:\n\n{previous_output}"
)

step3_tpl = PromptTemplate(
    "Based on this story outline, write a compelling one-paragraph opening "
    "hook that would draw a reader in:\n\n{previous_output}"
)


def step1(vars_, llm_fn=None):
    prompt = step1_tpl.render(**vars_)
    return llm_fn(prompt) if llm_fn else simple_llm(prompt)


def step2(vars_, llm_fn=None):
    prompt = step2_tpl.render(**vars_)
    return llm_fn(prompt) if llm_fn else simple_llm(prompt)


def step3(vars_, llm_fn=None):
    prompt = step3_tpl.render(**vars_)
    return llm_fn(prompt) if llm_fn else simple_llm(prompt)


# ── Run the chain ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    topic = "a lighthouse keeper who discovers the light has been attracting something other than ships"

    print("=" * 60)
    print("LLM Toolkit — Prompt Chain Demo")
    print("=" * 60)
    print(f"Topic: {topic}\n")

    chain = PromptChain(
        steps=[step1, step2, step3],
        llm_fn=simple_llm,
    )

    final_output = chain.run(topic=topic)

    for i, step_record in enumerate(chain.history, 1):
        print(f"── Step {i} ──────────────────────────────────────────────")
        print(step_record["output"])
        print()

    print("── Final Opening Hook ───────────────────────────────────")
    print(final_output)
    print()
    print(f"Chain completed in {len(chain.history)} steps.")
