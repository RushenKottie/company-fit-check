"""Workflow construction and programmatic entrypoints."""

from langgraph.graph import END, StateGraph

from application.messages import build_assistant_message
from graph.node_names import WorkflowNodeName, WorkflowRouteName
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
from models.state import CompanyFitState
from infrastructure.mlflow_tracking import (
    activate_mlflow_tracking,
    deactivate_mlflow_tracking,
    finalize_mlflow_tracking,
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
    graph.add_node(WorkflowNodeName.ENTRY.value, entry_node)
    graph.add_node(
        WorkflowNodeName.EXTRACT_AND_MASK_CV.value,
        extract_and_mask_cv_node,
    )
    graph.add_node(
        WorkflowNodeName.VALIDATE_PII_MASKING.value,
        validate_pii_masking_node,
    )
    graph.add_node(WorkflowNodeName.SIMPLIFY_CV.value, simplify_cv_node)
    graph.add_node(
        WorkflowNodeName.INTERPRET_USER_INPUT.value,
        interpret_user_input_node,
    )
    graph.add_node(
        WorkflowNodeName.VALIDATE_USER_INPUT_INTERPRETATION.value,
        validate_user_input_interpretation_node,
    )
    graph.add_node(
        WorkflowNodeName.REFINE_COMPANY_SEARCH.value,
        refine_company_search_node,
    )
    graph.add_node(WorkflowNodeName.SEARCH_COMPANIES.value, search_companies_node)
    graph.add_node(WorkflowNodeName.SCORE_COMPANIES.value, score_companies_node)

    graph.set_entry_point(WorkflowNodeName.ENTRY.value)
    graph.add_conditional_edges(
        WorkflowNodeName.ENTRY.value,
        route_from_entry,
        {
            WorkflowNodeName.EXTRACT_AND_MASK_CV.value: (
                WorkflowNodeName.EXTRACT_AND_MASK_CV.value
            ),
            WorkflowNodeName.INTERPRET_USER_INPUT.value: (
                WorkflowNodeName.INTERPRET_USER_INPUT.value
            ),
            WorkflowNodeName.REFINE_COMPANY_SEARCH.value: (
                WorkflowNodeName.REFINE_COMPANY_SEARCH.value
            ),
            WorkflowRouteName.STOP.value: END,
        },
    )
    graph.add_edge(
        WorkflowNodeName.EXTRACT_AND_MASK_CV.value,
        WorkflowNodeName.VALIDATE_PII_MASKING.value,
    )
    graph.add_conditional_edges(
        WorkflowNodeName.VALIDATE_PII_MASKING.value,
        route_after_privacy_check,
        {
            WorkflowNodeName.SIMPLIFY_CV.value: WorkflowNodeName.SIMPLIFY_CV.value,
            WorkflowRouteName.STOP.value: END,
        },
    )
    graph.add_edge(
        WorkflowNodeName.SIMPLIFY_CV.value,
        WorkflowNodeName.INTERPRET_USER_INPUT.value,
    )
    graph.add_conditional_edges(
        WorkflowNodeName.INTERPRET_USER_INPUT.value,
        route_after_user_input_interpretation,
        {
            WorkflowNodeName.VALIDATE_USER_INPUT_INTERPRETATION.value: (
                WorkflowNodeName.VALIDATE_USER_INPUT_INTERPRETATION.value
            ),
            WorkflowRouteName.STOP.value: END,
        },
    )
    graph.add_conditional_edges(
        WorkflowNodeName.VALIDATE_USER_INPUT_INTERPRETATION.value,
        route_after_validation,
        {
            WorkflowNodeName.SEARCH_COMPANIES.value: (
                WorkflowNodeName.SEARCH_COMPANIES.value
            ),
            WorkflowRouteName.STOP.value: END,
        },
    )
    graph.add_conditional_edges(
        WorkflowNodeName.REFINE_COMPANY_SEARCH.value,
        route_after_company_search_refinement,
        {
            WorkflowNodeName.SEARCH_COMPANIES.value: (
                WorkflowNodeName.SEARCH_COMPANIES.value
            ),
            WorkflowRouteName.STOP.value: END,
        },
    )
    graph.add_conditional_edges(
        WorkflowNodeName.SEARCH_COMPANIES.value,
        route_after_company_search,
        {
            WorkflowNodeName.SCORE_COMPANIES.value: (
                WorkflowNodeName.SCORE_COMPANIES.value
            ),
            WorkflowRouteName.STOP.value: END,
        },
    )
    graph.add_edge(WorkflowNodeName.SCORE_COMPANIES.value, END)
    return graph.compile()


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
                    "assistant_message": build_assistant_message(result),
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

    message = build_assistant_message(state).strip()
    if message:
        return message
    if state.get("session_status") == "running":
        return "Workflow is running."
    return None
