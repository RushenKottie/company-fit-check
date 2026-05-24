"""Workflow construction and programmatic entrypoints."""

from langgraph.graph import END, StateGraph

from interfaces.chainlit.presenters import (
    build_clarification_message,
    build_completion_message,
    build_failure_message,
)
from graph.nodes import (
    extract_and_mask_cv_node,
    interpret_user_input_node,
    refine_company_search_node,
    score_companies_node,
    search_companies_node,
    simplify_cv_node,
    validate_user_input_interpretation_node,
    validate_pii_masking_node,
)
from graph.routing import (
    route_after_company_search,
    route_after_company_search_refinement,
    route_after_user_input_interpretation,
    route_after_privacy_check,
    route_after_validation,
    route_from_entry,
)
from logging_utils import get_logger
from llm.client import is_guardrail_rephrase_message
from models.input import UserInput
from models.state import CompanyFitState, CompanySearchCriteria
from services.mlflow_tracking import (
    activate_mlflow_tracking,
    deactivate_mlflow_tracking,
    finalize_mlflow_tracking,
    log_clarification_answer_for_run,
    traced_operation,
    update_current_trace_session,
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
    graph.add_node("interpret_user_input", interpret_user_input_node)
    graph.add_node("validate_user_input_interpretation", validate_user_input_interpretation_node)
    graph.add_node("refine_company_search", refine_company_search_node)
    graph.add_node("search_companies", search_companies_node)
    graph.add_node("score_companies", score_companies_node)

    graph.set_entry_point("entry")
    graph.add_conditional_edges(
        "entry",
        route_from_entry,
        {
            "extract_and_mask_cv": "extract_and_mask_cv",
            "interpret_user_input": "interpret_user_input",
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
    graph.add_edge("simplify_cv", "interpret_user_input")
    graph.add_conditional_edges(
        "interpret_user_input",
        route_after_user_input_interpretation,
        {
            "validate_user_input_interpretation": "validate_user_input_interpretation",
            "stop": END,
        },
    )
    graph.add_conditional_edges(
        "validate_user_input_interpretation",
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


def create_initial_state(user_input: UserInput, *, run_id: str) -> CompanyFitState:
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
        latest_user_message_text=user_input.prompt,
        latest_user_message_kind="prompt",
        guardrail_rephrase_source=None,
        run_id=run_id,
        user_input_interpretation_clarification_iterations=0,
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
            "run_id": state.get("run_id"),
            "session_status": state.get("session_status"),
            "clarification_target": state.get("clarification_target"),
            "user_message": _build_request_preview(state),
            "user_message_kind": state.get("latest_user_message_kind"),
            "initial_prompt": (
                state.get("input").prompt if state.get("input") is not None else None
            ),
        },
        attributes={"app.name": "company-fit-check"},
    ) as span:
        update_current_trace_session(
            session_id=state.get("run_id"),
            request_preview=_build_request_preview(state),
        )
        try:
            result = app.invoke(state)
        finally:
            finalize_mlflow_tracking(result)
            deactivate_mlflow_tracking(tracking_context)
        update_current_trace_session(
            session_id=result.get("run_id") or state.get("run_id"),
            response_preview=_build_response_preview(result),
        )
        if span is not None:
            span.set_outputs(
                {
                    "session_status": result.get("session_status"),
                    "error": result.get("error"),
                    "company_count": len(result.get("companies", [])),
                    "score_count": len(result.get("company_scores", [])),
                    "clarification_target": result.get("clarification_target"),
                    "assistant_message": _build_assistant_message(result),
                }
            )
    logger.info("Workflow finished status=%s", result.get("session_status"))
    return result


def _build_request_preview(state: CompanyFitState) -> str | None:
    """Return the best short user-facing request preview for the session trace."""

    latest_user_message = (state.get("latest_user_message_text") or "").strip()
    if latest_user_message:
        return latest_user_message
    user_input = state.get("input")
    if user_input is None:
        return None
    prompt = user_input.prompt.strip()
    return prompt or None


def _build_response_preview(state: CompanyFitState) -> str | None:
    """Return the best short assistant-facing response preview for the session trace."""

    message = _build_assistant_message(state).strip()
    if message:
        return message
    if state.get("session_status") == "running":
        return "Workflow is running."
    return None


def _build_assistant_message(state: CompanyFitState) -> str:
    """Return the rendered assistant message for the current workflow state."""

    status = state.get("session_status")
    if status == "needs_clarification":
        return build_clarification_message(state)
    if status == "failed":
        return build_failure_message(state)
    if status == "completed":
        return build_completion_message(state)
    return ""


def apply_clarification(
    state: CompanyFitState,
    user_response: str,
) -> CompanyFitState:
    """Resume user-input interpretation after the user provides clarification."""

    logger.info(
        "Applying clarification target=%s response_length=%s",
        state.get("clarification_target"),
        len(user_response),
    )
    is_guardrail_rephrase = (
        state.get("clarification_target") == "user_input_interpretation"
        and is_guardrail_rephrase_message(state.get("pending_clarification_message"))
    )
    if state.get("clarification_target") == "company_search":
        state["company_search_clarification_iterations"] = (
            state.get("company_search_clarification_iterations", 0) + 1
        )
    elif not is_guardrail_rephrase:
        state["user_input_interpretation_clarification_iterations"] = (
            state.get("user_input_interpretation_clarification_iterations", 0) + 1
        )
    log_clarification_answer_for_run(
        run_id=state.get("run_id"),
        question=state.get("pending_clarification_message"),
        answer=user_response,
        target=state.get("clarification_target"),
    )
    if is_guardrail_rephrase:
        state["input"].prompt = user_response.strip()
        state["latest_clarification_response"] = None
        state["latest_user_message_text"] = user_response.strip()
        state["latest_user_message_kind"] = "prompt"
    else:
        state["latest_clarification_response"] = _build_clarification_context(
            state.get("pending_clarification_message"),
            user_response,
        )
        state["latest_user_message_text"] = user_response.strip()
        state["latest_user_message_kind"] = "clarification"
    state["session_status"] = "running"
    state["pending_clarification_message"] = None
    state["guardrail_rephrase_source"] = None
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
