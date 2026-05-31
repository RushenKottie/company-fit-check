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
            "Use the provided rubric text exactly as the scoring basis. Judge the "
            "agent's process before judging the final company list. First decide "
            "whether the initial prompt was sufficient, partially specified, or "
            "underspecified for company search and scoring. Distinguish concrete, "
            "checkable user requirements from subjective recommendation quality. "
            "Concrete requirements include requested number of companies, required "
            "region, required industry, company size, company stage, founding period, "
            "remote or on-site requirement, excluded categories, required output "
            "fields, or other explicit filters that can be checked directly. Pay "
            "close attention to whether the agent followed those concrete "
            "requirements. Do not treat measurable requirements as optional "
            "preferences. If the user asked for a specific amount, category, "
            "location, timeframe, or other hard condition, use that as important "
            "evidence when judging alignment and process quality. Do not let broadly "
            "acceptable final recommendations excuse a missed clarification or silent "
            "assumption when the process was weak. Also do not let subjective result "
            "imperfections dominate when the process was reasonable and explicit "
            "constraints were not ignored. Use the anchored "
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
                    "proceeding with company search and scoring. First classify the "
                    "initial prompt. A sufficient prompt gives enough explicit goals, "
                    "constraints, preferences, or evaluation axes to proceed without "
                    "more questions. A partially specified prompt gives some direction "
                    "but leaves important priorities open. An underspecified prompt asks "
                    "for broad CV matching without meaningful preferences, axes, or "
                    "constraints. For partially specified or underspecified prompts, the "
                    "agent should either ask a concise, relevant clarification question "
                    "or clearly state the assumptions it will use and invite the user to "
                    "add priorities before or while proceeding. The exact wording does "
                    "not matter. Do not require clarification for its own sake when the "
                    "prompt and CV already make the user's intent clear enough. Penalize "
                    "unnecessary clarification that adds friction despite a sufficient "
                    "prompt. Do not use a broadly acceptable final CSV to excuse a missed "
                    "clarification step when the initial prompt lacked important "
                    "decision criteria. Score 1 when the prompt was clearly "
                    "underspecified and the agent proceeded as if key preferences were "
                    "known, or asked irrelevant/unhelpful questions. Score 2 when "
                    "clarification was noticeably weak, incomplete, excessive, or poorly "
                    "targeted. Score 3 when the clarification decision was acceptable: "
                    "the agent asked a useful question, made reasonable assumptions "
                    "transparent, or correctly skipped questions because the prompt was "
                    "sufficient. Score 4 when the agent matched its clarification "
                    "behavior well to the amount of information provided, with only "
                    "minor gaps. Score 5 when clarification behavior was concise, "
                    "well-timed, and clearly improved or appropriately streamlined the "
                    "process."
                ),
            ),
            JudgeMetricDefinition(
                name="assumption_control",
                rubric=(
                    "Assumption Control. Evaluate whether the agent avoided making "
                    "unsupported assumptions about the user's goals, constraints, "
                    "preferences, or evaluation axes. Focus on process transparency. "
                    "When the prompt lacks explicit axes or preferences, CV-derived "
                    "priorities can be a reasonable starting point only if the agent "
                    "treats them as assumptions or provisional defaults rather than as "
                    "known user preferences. Penalize silent invention of important "
                    "regions, industries, company types, constraints, seniority needs, "
                    "or scoring priorities. Do not penalize minor low-risk defaults, "
                    "basic interpretations of the CV, or assumptions that are clearly "
                    "presented as provisional. Do not let a decent final company list "
                    "erase unsupported assumptions made earlier in the process. Score 1 "
                    "when the agent invented key goals, constraints, preferences, axes, "
                    "or facts without evidence and presented them as settled. Score 2 "
                    "when unsupported or opaque assumptions are noticeable and reduce "
                    "trust in the process. Score 3 when assumptions were limited, "
                    "understandable, or mostly transparent enough for a user to correct. "
                    "Score 4 when the agent mostly stayed grounded and handled missing "
                    "information with clear, low-risk assumptions. Score 5 when the "
                    "agent stayed tightly grounded in the provided evidence and handled "
                    "missing information explicitly without meaningful unsupported "
                    "leaps."
                ),
            ),
            JudgeMetricDefinition(
                name="reasoning_relevance_constraint_alignment",
                rubric=(
                    "Reasoning Relevance & Constraint Alignment. Evaluate whether the "
                    "generated company recommendations and scoring rationale are logically "
                    "consistent with the user's profile, goals, explicitly stated "
                    "constraints, and any axes the agent used. Treat this metric as a "
                    "secondary sanity check on the result, not as a search-quality "
                    "competition. Do not evaluate whether these are objectively the best "
                    "possible companies. Do not penalize merely generic, incomplete, or "
                    "subjectively imperfect recommendations if they are still plausible "
                    "and do not contradict the user's explicit request. Penalize clear "
                    "mismatches, ignored explicit constraints, incoherent rationale, "
                    "false claims, or discovery reasons that materially misstate why a "
                    "company fits. If the initial prompt was underspecified, do not "
                    "punish the result simply because it used broad or CV-derived "
                    "criteria; judge that process mainly under clarification_quality and "
                    "assumption_control. Do not infer additional unstated preferences "
                    "beyond explicitly provided constraints. Score 1 when there are "
                    "clear mismatches, ignored explicit constraints, contradictions, or "
                    "false justifications that materially break alignment. Score 2 when "
                    "the output has noticeable relevance or constraint problems that go "
                    "beyond subjective quality concerns. Score 3 when the "
                    "recommendations are broadly plausible and aligned despite generic "
                    "reasoning or imperfections. Score 4 when recommendations and "
                    "rationale are clearly relevant, coherent, and constraint-aware with "
                    "only minor weaknesses. Score 5 when recommendations, scoring "
                    "rationale, and discovery reasons are consistently precise, "
                    "relevant, and tightly aligned with the user's stated goals and "
                    "constraints."
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
        "- Before scoring, classify the initial prompt as sufficient, partially specified, or underspecified.\n"
        "- Use that classification to judge whether the agent's clarification behavior was appropriate.\n"
        "- Do not use a broadly acceptable final CSV to excuse weak clarification or silent assumptions.\n"
        "- Do not over-penalize subjective result imperfections when the process was reasonable.\n"
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
