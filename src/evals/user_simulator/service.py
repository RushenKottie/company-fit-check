"""Stateless concurrent user simulator for runner-driven eval sessions."""

from __future__ import annotations

from collections.abc import Mapping

from evals.nondeterministic.case_models import NondeterministicCase
from evals.user_simulator.models import (
    ConversationTurn,
    ReplyToAgentRequest,
    ReplyToAgentResponse,
    StartCaseRequest,
    StartCaseResponse,
    UserSimulatorError,
)
from llm.client import generate_user_simulator_reply
from logging_utils import get_logger


logger = get_logger(__name__)


class UserSimulator:
    """Stateless runner-facing user simulator safe for concurrent calls."""

    def __init__(
        self,
        *,
        case_index: Mapping[int, NondeterministicCase],
    ) -> None:
        """Create a simulator backed by the provided case index."""

        self._case_index = case_index

    def start_case(self, request: StartCaseRequest) -> StartCaseResponse:
        """Return the first prompt and PDF path for one case."""

        case = self._get_case(request.case_id)
        if not case.first_prompt.strip():
            raise UserSimulatorError(
                "case_missing_first_prompt",
                f"Case {case.id} is missing first_prompt.",
            )
        if not case.pdf_path.strip():
            raise UserSimulatorError(
                "case_missing_pdf_path",
                f"Case {case.id} is missing pdf_path.",
            )

        return StartCaseResponse(
            case_id=case.id,
            case_name=case.name,
            prompt=case.first_prompt.strip(),
            pdf_path=case.pdf_path.strip(),
        )

    def reply_to_agent(self, request: ReplyToAgentRequest) -> ReplyToAgentResponse:
        """Return one plain-text user reply for the latest agent message."""

        case = self._get_case(request.case_id)
        agent_message = request.agent_message
        if not agent_message.strip():
            raise UserSimulatorError(
                "empty_agent_message",
                "reply_to_agent requires a non-empty agent_message.",
            )

        try:
            answer = generate_user_simulator_reply(
                self._build_reply_prompt(case, agent_message, request.conversation)
            )
        except Exception as exc:
            logger.exception("User simulator LLM invocation failed case_id=%s", case.id)
            raise UserSimulatorError(
                "llm_invocation_failed",
                "User simulator failed to generate a reply.",
            ) from exc

        if answer is None:
            raise UserSimulatorError(
                "llm_not_configured",
                "User simulator Anthropic Foundry model is not configured.",
            )
        if not answer:
            raise UserSimulatorError(
                "empty_llm_response",
                "User simulator LLM returned an empty reply.",
            )

        return ReplyToAgentResponse(
            run_id=request.run_id,
            case_id=case.id,
            case_name=case.name,
            answer=answer,
        )

    def _get_case(self, case_id: int) -> NondeterministicCase:
        """Return one case or raise a typed error."""

        case = self._case_index.get(case_id)
        if case is None:
            raise UserSimulatorError(
                "unknown_case_id",
                f"Unknown non-deterministic case id: {case_id}.",
            )
        return case

    def _build_reply_prompt(
        self,
        case: NondeterministicCase,
        agent_message: str,
        conversation: list[ConversationTurn],
    ) -> str:
        """Build one text prompt for one follow-up reply."""

        behavioral_traits = "\n".join(
            f"- {trait}" for trait in case.communication_style.behavioral_traits
        )
        filter_criteria = "\n".join(f"- {criterion}" for criterion in case.filter_criteria)
        axes = "\n".join(f"- {axis}" for axis in case.axes)
        conversation_so_far = _format_conversation(conversation)

        return (
            "You are a user simulator. Write the next message the user would send in "
            "this conversation.\n\n"
            "Return only the user's next plain-text message. Do not explain, preface, "
            "quote, summarize, or say what you would answer. Start with the answer "
            "itself.\n\n"
            "Reply like the user, not like the agent. Stay consistent with the user's "
            "background, goals, constraints, and communication style. Do not invent "
            "unsupported facts.\n\n"
            "Priority rules:\n"
            "1. Answer the latest agent message directly.\n"
            "2. Keep visible conversation text separate from private case data: the "
            "initial prompt and transcript show what the user has said, while case "
            "data describes the user's intended background and preferences.\n"
            "3. Keep every reply strictly aligned with the initial prompt and the "
            "provided case data.\n"
            "4. Do not contradict, weaken, or drift away from the initial prompt's "
            "stated goals, regions, constraints, or priorities unless the agent's "
            "latest message gives a valid reason to narrow in on one part of them.\n"
            "5. If the agent asks for a detail that is already stated or strongly "
            "implied by the visible conversation or private case data, answer with "
            "only the grounded fact that addresses it.\n"
            "6. If the agent asks a forced-choice question and neither option is fully "
            "supported by the case data, do not pick one just to be helpful. Answer "
            "only what is grounded and briefly note what is not established.\n"
            "7. Be especially careful with employment status, visa mechanics, "
            "timeline, compensation, employer names, and relocation facts. Do not add "
            "specifics unless they are explicitly established.\n"
            "8. Do not turn the reply into a polished memo, cover letter, scripted "
            "interview answer, support message, coaching response, or survey answer.\n"
            "9. Keep the reply short to medium unless the agent asks for detail.\n\n"
            "Private case data:\n"
            f"Profession: {case.profession}\n"
            f"Experience: {case.experience}\n"
            f"Goal: {case.goal}\n"
            f"Relevant filters/preferences:\n{filter_criteria}\n"
            f"Axes:\n{axes}\n\n"
            "Visible initial message previously sent by the user:\n"
            f"{case.first_prompt}\n\n"
            "Visible conversation so far:\n"
            f"{conversation_so_far}\n\n"
            "Latest agent message to answer:\n"
            f"{agent_message}\n\n"
            "Tone and structure guidance:\n"
            f"Communication style: {case.communication_style.description}\n"
            f"Behavioral traits:\n{behavioral_traits}\n\n"
            "Use the style guidance for voice, rhythm, and structure."
        )


def _format_conversation(conversation: list[ConversationTurn]) -> str:
    """Return a compact transcript block for simulator grounding."""

    if not conversation:
        return "No prior turns were provided."

    lines: list[str] = []
    for turn in conversation:
        speaker = turn.speaker.strip().lower()
        label = "User" if speaker == "user" else "Agent"
        if turn.message:
            lines.append(f"{label}: {turn.message}")
    return "\n".join(lines) or "No prior turns were provided."
