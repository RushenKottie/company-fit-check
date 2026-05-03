"""Workflow construction and programmatic entrypoints."""

import uuid

from langgraph.graph import END, StateGraph

from company_fit_check.graph.nodes import (
    extract_and_mask_cv_node,
    interpret_prompt_node,
    refine_company_search_node,
    score_companies_node,
    search_companies_node,
    simplify_cv_node,
    validate_interpretation_node,
    validate_pii_masking_node,
)
from company_fit_check.graph.routing import (
    route_after_company_search,
    route_after_company_search_refinement,
    route_after_interpretation,
    route_after_privacy_check,
    route_after_validation,
    route_from_entry,
)
from company_fit_check.logging_utils import get_logger
from company_fit_check.models.input import UserInput
from company_fit_check.models.state import CompanyFitState, CompanySearchCriteria
from company_fit_check.services.mlflow_tracking import (
    activate_mlflow_tracking,
    deactivate_mlflow_tracking,
    finalize_mlflow_tracking,
    log_clarification_answer_for_run,
    traced_operation,
)
from mlflow.entities.span import SpanType

logger = get_logger(__name__)


def entry_node(state: CompanyFitState) -> CompanyFitState:
    """No-op entry point used to resume at the right boundary."""

    return state


def build_graph():
    """Build the minimal LangGraph workflow."""

    logger.info("Building workflow graph.")
    graph = StateGraph(CompanyFitState)
    graph.add_node("entry", entry_node)
    graph.add_node("extract_and_mask_cv", extract_and_mask_cv_node)
    graph.add_node("validate_pii_masking", validate_pii_masking_node)
    graph.add_node("simplify_cv", simplify_cv_node)
    graph.add_node("interpret_prompt", interpret_prompt_node)
    graph.add_node("validate_interpretation", validate_interpretation_node)
    graph.add_node("refine_company_search", refine_company_search_node)
    graph.add_node("search_companies", search_companies_node)
    graph.add_node("score_companies", score_companies_node)

    graph.set_entry_point("entry")
    graph.add_conditional_edges(
        "entry",
        route_from_entry,
        {
            "extract_and_mask_cv": "extract_and_mask_cv",
            "interpret_prompt": "interpret_prompt",
            "refine_company_search": "refine_company_search",
            "stop": END,
        },
    )
    graph.add_edge("extract_and_mask_cv", "validate_pii_masking")
    graph.add_conditional_edges(
        "validate_pii_masking",
        route_after_privacy_check,
        {
            "simplify_cv": "simplify_cv",
            "stop": END,
        },
    )
    graph.add_edge("simplify_cv", "interpret_prompt")
    graph.add_conditional_edges(
        "interpret_prompt",
        route_after_interpretation,
        {
            "validate_interpretation": "validate_interpretation",
            "stop": END,
        },
    )
    graph.add_conditional_edges(
        "validate_interpretation",
        route_after_validation,
        {
            "search_companies": "search_companies",
            "stop": END,
        },
    )
    graph.add_conditional_edges(
        "refine_company_search",
        route_after_company_search_refinement,
        {
            "search_companies": "search_companies",
            "stop": END,
        },
    )
    graph.add_conditional_edges(
        "search_companies",
        route_after_company_search,
        {
            "score_companies": "score_companies",
            "stop": END,
        },
    )
    graph.add_edge("score_companies", END)
    return graph.compile()


def create_initial_state(user_input: UserInput) -> CompanyFitState:
    """Create fresh in-memory state for one workflow run."""

    logger.info(
        "Creating initial workflow state prompt_length=%s pdf_bytes=%s",
        len(user_input.prompt),
        len(user_input.cv_pdf_bytes),
    )
    return CompanyFitState(
        input=user_input,
        masked_cv_text=None,
        pii_masking_status="not_started",
        simplified_cv_text=None,
        company_search_criteria=CompanySearchCriteria(),
        axes=[],
        companies=[],
        company_scores=[],
        pending_clarification_message=None,
        latest_clarification_response=None,
        clarification_target=None,
        debug_session_id=uuid.uuid4().hex,
        mlflow_run_id=None,
        interpretation_clarification_iterations=0,
        company_search_clarification_iterations=0,
        session_status="running",
        error=None,
    )


def run_workflow(state: CompanyFitState) -> CompanyFitState:
    """Run the graph until it completes, fails, or needs user clarification."""

    logger.info("Invoking workflow status=%s", state.get("session_status"))
    app = build_graph()
    tracking_context = activate_mlflow_tracking(state)
    result = state
    with traced_operation(
        "workflow.run",
        span_type=SpanType.WORKFLOW,
        inputs={
            "session_id": state.get("debug_session_id"),
            "run_id": state.get("mlflow_run_id"),
            "session_status": state.get("session_status"),
            "clarification_target": state.get("clarification_target"),
            "prompt": state.get("input").prompt if state.get("input") is not None else None,
        },
        attributes={"app.name": "company-fit-check"},
    ) as span:
        try:
            result = app.invoke(state)
        finally:
            finalize_mlflow_tracking(result)
            deactivate_mlflow_tracking(tracking_context)
        if span is not None:
            span.set_outputs(
                {
                    "session_status": result.get("session_status"),
                    "error": result.get("error"),
                    "company_count": len(result.get("companies", [])),
                    "score_count": len(result.get("company_scores", [])),
                    "clarification_target": result.get("clarification_target"),
                }
            )
    logger.info("Workflow finished status=%s", result.get("session_status"))
    return result


def apply_clarification(
    state: CompanyFitState,
    user_response: str,
) -> CompanyFitState:
    """Resume interpretation after the user provides clarification."""

    logger.info(
        "Applying clarification target=%s response_length=%s",
        state.get("clarification_target"),
        len(user_response),
    )
    if state.get("clarification_target") == "company_search":
        state["company_search_clarification_iterations"] = (
            state.get("company_search_clarification_iterations", 0) + 1
        )
    else:
        state["interpretation_clarification_iterations"] = (
            state.get("interpretation_clarification_iterations", 0) + 1
        )
    log_clarification_answer_for_run(
        run_id=state.get("mlflow_run_id"),
        question=state.get("pending_clarification_message"),
        answer=user_response,
        target=state.get("clarification_target"),
    )
    state["latest_clarification_response"] = _build_clarification_context(
        state.get("pending_clarification_message"),
        user_response,
    )
    state["session_status"] = "running"
    state["pending_clarification_message"] = None
    return run_workflow(state)


def _build_clarification_context(
    assistant_question: str | None,
    user_response: str,
) -> str:
    """Combine the clarification question and answer into one scoped context block."""

    question = (assistant_question or "").strip()
    answer = user_response.strip()
    if not question:
        return answer

    return (
        "Previous information was not enough, so the model requested clarification.\n"
        f"Agent question: {question}\n"
        f"User answer: {answer}"
    )
