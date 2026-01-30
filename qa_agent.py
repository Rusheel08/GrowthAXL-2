from typing import Optional, Dict, List
from langfuse import Langfuse

from groq_client import GroqClient
from groq_judge import GroqJudge
from config import (
    LANGFUSE_ENABLED,
    GROQ_API_KEY,
    CHAT_MODEL,
    JUDGE_MODEL,
)


class QAAgent:
    def __init__(self):
        # Main LLM
        self.client = GroqClient(
            api_key=GROQ_API_KEY,
            model=CHAT_MODEL,
        )

        # LLM-as-a-Judge
        self.judge = GroqJudge(
            api_key=GROQ_API_KEY,
            model=JUDGE_MODEL,
        )

        # Langfuse
        self.lf = Langfuse() if LANGFUSE_ENABLED else None

    # Helpers

    @staticmethod
    def _classify_output(answer: str) -> str:
        if not answer or not answer.strip():
            return "empty"
        if "def " in answer or "class " in answer or "```" in answer:
            return "code"
        return "natural_language"

    # Main execution

    def run(
        self,
        question: str,
        reference_context: Optional[str] = None,
        is_factual_task: bool = True,
        has_gold_answer: bool = False,
    ) -> str:

        # 1. Create trace
        trace = None
        if self.lf:
            trace = self.lf.trace(
                name="chat.request",
                input={"question": question},
            )
            trace.update(
                metadata={
                    "task_type": "qa",
                    "has_reference_context": bool(reference_context),
                    "has_gold_answer": has_gold_answer,
                }
            )

        # 2. Generate answer
        answer = self.client.generate(
            messages=[{"role": "user", "content": question}]
        )

        # 3. Classify output
        output_type = self._classify_output(answer)
        has_output = bool(answer and answer.strip())

        evaluators_run: List[str] = []
        evaluators_skipped: List[str] = []

        if has_output and output_type == "natural_language":
            evaluators_run.extend(["relevance", "toxicity"])
        else:
            evaluators_skipped.extend(["relevance", "toxicity"])

        if has_output and reference_context and is_factual_task:
            evaluators_run.extend(["hallucination", "faithfulness"])
        else:
            evaluators_skipped.extend(["hallucination", "faithfulness"])

        if has_gold_answer:
            evaluators_run.append("correctness")
        else:
            evaluators_skipped.append("correctness")

        if trace:
            trace.update(
                metadata={
                    "output_type": output_type,
                    "evaluators_run": evaluators_run,
                    "evaluators_skipped": evaluators_skipped,
                }
            )

        # 4. Run judge ONCE
        if trace and has_output:
            metrics: Dict[str, float] = self.judge.evaluate(
                question=question,
                answer=answer,
                reference_context=reference_context,
                is_factual_task=is_factual_task,
                has_gold_answer=has_gold_answer,
            )

            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    trace.score(
                     name=name,
                     value=value
                )       

        # 5. Finalize trace
        if trace:
            trace.update(output={"answer": answer})

        return answer