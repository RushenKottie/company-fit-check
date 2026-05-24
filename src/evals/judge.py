"""LLM-as-judge helpers for non-deterministic regression runs."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType

from evals.models import (
    JudgeMetricDefinition,
    RegressionJudgeConfig,
    RegressionJudgeRequest,
    RegressionJudgeResponse,
)
from llm.client import create_llm_judge_chat_model
from logging_utils import get_logger
from services.mlflow_tracking import (
    log_llm_prompt_artifact,
    traced_operation,
    update_current_trace_session,
)

logger = get_logger(__name__)


def get_default_regression_judge_config() -> RegressionJudgeConfig:
    """Return the default prompt and metric definitions for judge scoring."""

    return RegressionJudgeConfig(
        system_prompt=(
            "You are an expert evaluator for company-fit regression cases. "
            "Review the provided conversation transcript, generated CSV, initial prompt, "
            "raw CV text, and metric definitions. Score only the requested metrics. "
            "Use the provided rubric text exactly as the scoring basis. Use the anchored "
            "1 to 5 scale consistently across all metrics: 1 means clear failure, 2 means "
            "weak, 3 means acceptable, 4 means strong, and 5 means excellent. Be "
            "evidence-led, conservative, and concise. Use simple, clear, plain language "
            "in all optional reasons and notes. Prefer short sentences. Avoid jargon, "
            "inflated wording, and abstract evaluator language. Explain scores in a way "
            "that is quick to understand. Do not write long essays. Return structured "
            "output only."
        ),
        metrics=[
            JudgeMetricDefinition(
                name="clarification_quality",
                rubric=(
                    "Clarification Quality. Evaluate whether the agent asked "
                    "clarification questions when they were materially needed before "
                    "proceeding with company search and scoring. Focus on whether "
                    "important missing information was clarified when its absence would "
                    "materially reduce recommendation quality, whether clarification "
                    "questions were relevant to the user's stated goals and constraints, "
                    "whether the agent avoided excessive or unnecessary questioning, and "
                    "whether the agent gathered enough information before making "
                    "recommendations. Do not penalize the agent for not asking "
                    "clarification questions if the initial prompt and CV already "
                    "provide enough information to produce broadly relevant and "
                    "constraint-aligned recommendations. Reasonable best-effort progress "
                    "without clarification is acceptable when missing details are "
                    "non-critical. Only penalize lack of clarification when the missing "
                    "information materially affects recommendation quality or causes "
                    "clear downstream mismatches. When in doubt, use the final "
                    "recommendation quality and constraint alignment as evidence of "
                    "whether missing clarification was actually harmful. Score 1 when "
                    "important clarification was clearly missed and that caused harmful "
                    "downstream fit problems. Score 2 when clarification was noticeably "
                    "weak or incomplete and this made the result unreliable. Score 3 "
                    "when clarification was basically sufficient, or correctly skipped "
                    "because the prompt and CV were already adequate. Score 4 when the "
                    "agent asked the right clarification questions with only minor "
                    "non-impactful gaps. Score 5 when clarification behavior was precise, "
                    "minimal, and clearly improved recommendation quality."
                ),
            ),
            JudgeMetricDefinition(
                name="assumption_control",
                rubric=(
                    "Assumption Control. Evaluate whether the agent avoided making "
                    "unsupported assumptions about the user's goals, constraints, or "
                    "preferences. Focus on whether the agent invented missing "
                    "information instead of clarifying it, whether the agent assumed "
                    "important regions, industries, priorities, or constraints without "
                    "evidence, whether those assumptions materially affected "
                    "recommendation quality, and whether the agent used reasonable "
                    "low-risk defaults when needed. Do not penalize minor or low-risk "
                    "assumptions that do not materially change the final "
                    "recommendations. If the final recommendations remain broadly "
                    "relevant and aligned with the user's stated goals and constraints, "
                    "score this metric more leniently. Only penalize strongly when "
                    "unsupported assumptions create clear mismatches, contradictions, or "
                    "ignored constraints in the final output. When in doubt, use the "
                    "final recommendation quality and constraint alignment as evidence "
                    "of whether assumptions were actually harmful. Score 1 when the "
                    "agent invented key goals, constraints, preferences, or facts "
                    "without evidence and that materially harmed the output. Score 2 "
                    "when unsupported assumptions are noticeable and meaningfully reduce "
                    "trust or quality. Score 3 when the agent made some assumptions but "
                    "they were limited, understandable, and not materially harmful. "
                    "Score 4 when the agent mostly avoided unsupported assumptions and "
                    "used reasonable low-risk defaults when needed. Score 5 when the "
                    "agent stayed tightly grounded in the provided evidence and handled "
                    "missing information without making meaningful unsupported leaps."
                ),
            ),
            JudgeMetricDefinition(
                name="reasoning_relevance_constraint_alignment",
                rubric=(
                    "Reasoning Relevance & Constraint Alignment. Evaluate whether the "
                    "generated company recommendations and scoring rationale are logically "
                    "consistent with the user's profile, goals, explicitly stated "
                    "constraints, and selected evaluation axes. The evaluation should "
                    "focus on whether recommended companies generally match the requested "
                    "region, domain, or industry, whether explicitly stated filtering "
                    "criteria are respected, whether the reasoning sounds coherent and "
                    "relevant, whether assigned scores are reasonably explained, and "
                    "whether there are obvious mismatches, contradictions, or ignored "
                    "constraints. Also evaluate whether each company's discovery_reason "
                    "is consistent with the user's stated request and with the company "
                    "information in the CSV. Penalize discovery reasons that introduce "
                    "false, unsupported, contradictory, or clearly irrelevant claims "
                    "about why the company matches the request. If discovery_reason "
                    "contains clear false information or materially misstates why the "
                    "company fits the user's request, this should substantially reduce "
                    "the reasoning_relevance_constraint_alignment score. A weak or "
                    "generic discovery_reason should be penalized only when it materially "
                    "misrepresents fit or hides a mismatch. Focus on detecting clear "
                    "mismatches, contradictions, false justifications, or ignored "
                    "constraints rather than ranking recommendation quality. Do not "
                    "evaluate whether these are objectively the best possible companies. "
                    "Do not penalize recommendations for being suboptimal if they are "
                    "still reasonable and relevant. Alternative but reasonable "
                    "recommendations should not be penalized. Minor imperfections and "
                    "alternative interpretations are acceptable. Do not infer additional "
                    "unstated preferences beyond explicitly provided constraints. The "
                    "goal is to determine whether the recommendations are broadly aligned "
                    "with the user's stated constraints and goals, not to identify the "
                    "best possible companies. Score 1 when there are clear mismatches, "
                    "ignored constraints, contradictions, or false justifications that "
                    "materially break alignment. Score 2 when the output has noticeable "
                    "relevance or constraint problems and feels weak or unreliable. "
                    "Score 3 when the recommendations are broadly reasonable and aligned "
                    "despite some imperfections. Score 4 when the recommendations and "
                    "rationale are clearly relevant, coherent, and constraint-aware with "
                    "only minor weaknesses. Score 5 when the recommendations, scoring "
                    "rationale, and discovery reasons are consistently precise, relevant, "
                    "and tightly aligned with the user's stated goals and constraints."
                ),
            ),
        ],
    )


def judge_regression_case(request: RegressionJudgeRequest) -> RegressionJudgeResponse:
    """Call the dedicated LLM judge for one regression case."""

    llm = create_llm_judge_chat_model()
    if llm is None:
        raise RuntimeError("Azure OpenAI is not configured for LLM judge scoring.")

    structured_llm = llm.with_structured_output(RegressionJudgeResponse)
    with traced_operation(
        "llm.judge_regression_case",
        span_type=SpanType.EVALUATOR,
        inputs=request.model_dump(),
    ) as span:
        update_current_trace_session(
            session_id=request.run_id,
            request_preview=f"Judge {request.case_name}",
        )
        messages = [
            SystemMessage(content=request.judge_system_prompt),
            HumanMessage(content=_build_judge_prompt(request)),
        ]
        log_llm_prompt_artifact("llm-prompt-judge-regression-case", messages)
        result = structured_llm.invoke(messages)
        update_current_trace_session(
            session_id=request.run_id,
            response_preview="Judge scoring completed.",
        )
        if span is not None:
            span.set_outputs(result.model_dump())
        return result


def _build_judge_prompt(request: RegressionJudgeRequest) -> str:
    """Build the case material payload for the regression judge."""

    metric_block = "\n".join(
        f"- {metric.name}: {metric.rubric}" for metric in request.judge_metrics
    )
    return (
        "Judge system prompt:\n"
        f"{request.judge_system_prompt}\n\n"
        "Metric definitions and rubrics:\n"
        f"{metric_block}\n\n"
        "Judge instructions:\n"
        "- Score exactly the three named metrics.\n"
        "- Use only the provided metric definitions and evidence in the case material.\n"
        "- Return each metric as an integer from 1 to 5.\n"
        "- Use this anchored scale consistently: 1=clear failure, 2=weak, 3=acceptable, 4=strong, 5=excellent.\n"
        "- Provide short optional reasons when helpful.\n"
        "- Write reasons in simple, direct, easy-to-understand language.\n"
        "- Keep each reason brief and concrete.\n"
        "- Avoid formal evaluation jargon unless necessary.\n\n"
        "Case metadata:\n"
        f"- case_id: {request.case_id}\n"
        f"- case_name: {request.case_name}\n"
        f"- run_id: {request.run_id}\n"
        f"- workflow_status: {request.status}\n\n"
        "Initial prompt:\n"
        f"{request.initial_prompt}\n\n"
        "Raw CV text:\n"
        f"{request.unmasked_cv_text}\n\n"
        "Generated CSV:\n"
        f"{request.generated_csv}\n\n"
        "Transcript JSON:\n"
        f"{request.transcript_json}"
    )
