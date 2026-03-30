import os
import anthropic
import pandas as pd
from config import CLAUDE_MODEL


def _get_api_key() -> str | None:
    """Read API key from Streamlit secrets (cloud) or environment (local)."""
    try:
        import streamlit as st

        return st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


from ai.prompts import ANALYST_PERSONA, STAGE_PROMPTS
from ai.context_builder import (
    build_dataset_context,
    build_model_results_context,
    build_optimization_context,
)  # noqa: F401 — ab_test/causal used by new methods below


class DataAnalyst:
    """Claude-powered data science advisor.

    Maintains conversation history so the AI remembers previous stages
    within a session (what was cleaned, what was engineered, etc.).
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=_get_api_key())
        self.history: list[dict] = []

    # ── Core API call ─────────────────────────────────────────────────────────

    def _chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=ANALYST_PERSONA,
            messages=self.history,
        )
        reply = response.content[0].text
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self.history = []

    # ── Stage-specific analysis calls ─────────────────────────────────────────

    def analyze_intent(self, df: pd.DataFrame, user_goal: str) -> str:
        """Stage 0: analyze the uploaded dataset + user intent, propose a workflow."""
        ctx = build_dataset_context(df)
        prompt = STAGE_PROMPTS["intent"] + f"\n\n{ctx}"
        if user_goal.strip():
            prompt += f"\n\n**User's stated goal:** {user_goal}"
        else:
            prompt += "\n\n**User's stated goal:** Not specified — propose the most useful starting point."
        return self._chat(prompt)

    def analyze_cleaning(self, df: pd.DataFrame, meta: dict) -> str:
        """Stage 2: review data quality, recommend cleaning strategy."""
        ctx = build_dataset_context(df, meta)
        prompt = STAGE_PROMPTS["clean"] + f"\n\n{ctx}"
        return self._chat(prompt)

    def analyze_eda(self, df: pd.DataFrame, meta: dict) -> str:
        """Stage 3: interpret EDA results."""
        ctx = build_dataset_context(df, meta)
        prompt = STAGE_PROMPTS["eda"] + f"\n\n{ctx}"
        return self._chat(prompt)

    def analyze_features(self, df: pd.DataFrame, meta: dict) -> str:
        """Stage 4: guide feature engineering decisions."""
        ctx = build_dataset_context(df, meta)
        prompt = STAGE_PROMPTS["feature_eng"] + f"\n\n{ctx}"
        return self._chat(prompt)

    def analyze_models(self, leaderboard: pd.DataFrame, meta: dict) -> str:
        """Stage 5: interpret model benchmarking leaderboard."""
        ctx = build_model_results_context(leaderboard, meta.get("task_type", "unknown"))
        prompt = STAGE_PROMPTS["model_select"] + f"\n\n{ctx}"
        return self._chat(prompt)

    def analyze_optimization(
        self, best_params: dict, baseline: float, best_score: float, model_name: str
    ) -> str:
        """Stage 6: interpret hyperparameter optimization results."""
        ctx = build_optimization_context(best_params, baseline, best_score, model_name)
        prompt = STAGE_PROMPTS["optimize"] + f"\n\n{ctx}"
        return self._chat(prompt)

    def generate_report_summary(self, meta: dict) -> str:
        """Stage 7: executive summary for the final report."""
        context_lines = ["**Full session summary:**"]
        for k, v in meta.items():
            if v is not None:
                context_lines.append(f"- {k}: {v}")
        prompt = STAGE_PROMPTS["report"] + "\n\n" + "\n".join(context_lines)
        return self._chat(prompt)

    def analyze_ab_test(self, results: dict) -> str:
        """Interpret A/B test results."""
        from ai.context_builder import build_ab_test_context

        ctx = build_ab_test_context(results)
        prompt = STAGE_PROMPTS["ab_test"] + f"\n\n{ctx}"
        return self._chat(prompt)

    def analyze_causal(self, results: dict) -> str:
        """Interpret causal inference results."""
        from ai.context_builder import build_causal_context

        ctx = build_causal_context(results)
        prompt = STAGE_PROMPTS["causal"] + f"\n\n{ctx}"
        return self._chat(prompt)

    def chat(self, user_message: str) -> str:
        """Free-form chat — user can ask anything about their data at any point."""
        return self._chat(user_message)
