"""Stateless concurrent user simulator for runner-driven sessions."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from evals.non_deterministic_case_loader import build_non_deterministic_case_index
from evals.non_deterministic_models import NonDeterministicRegressionCase
from llm.client import create_user_simulator_chat_model
from logging_utils import get_logger
from user_simulator.models import (
    ConversationTurn,
    ReplyToAgentRequest,
    ReplyToAgentResponse,
    StartCaseRequest,
    StartCaseResponse,
    UserSimulatorError,
)


logger = get_logger(__name__)


class _SupportsInvoke(Protocol):
    """Minimal protocol for injected chat models."""

    def invoke(self, messages: list[object]) -> object:
        """Return one chat completion object."""


@lru_cache(maxsize=1)
def load_cached_cases_by_id() -> Mapping[int, NonDeterministicRegressionCase]:
    """Load the case JSON files once, build a read-only ``case_id -> case`` map, and reuse it."""

    return build_non_deterministic_case_index()


def list_available_case_ids() -> list[int]:
    """Return the sorted non-deterministic regression case ids."""

    return sorted(load_cached_cases_by_id())


class UserSimulator:
    """Stateless runner-facing user simulator safe for concurrent calls."""

    def __init__(
        self,
        *,
        case_index: Mapping[int, NonDeterministicRegressionCase] | None = None,
        llm: _SupportsInvoke | None = None,
    ) -> None:
        self._case_index = case_index if case_index is not None else load_cached_cases_by_id()
        self._llm = llm

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
        agent_message = request.agent_message.strip()
        if not agent_message:
            raise UserSimulatorError(
                "empty_agent_message",
                "reply_to_agent requires a non-empty agent_message.",
            )

        llm = self._llm or create_user_simulator_chat_model()
        if llm is None:
            raise UserSimulatorError(
                "llm_not_configured",
                "User simulator Anthropic Foundry model is not configured.",
            )

        messages = self._build_reply_messages(case, agent_message, request.conversation)
        try:
            response = llm.invoke(messages)
        except Exception as exc:  # pragma: no cover - exercised via unit tests with fakes
            logger.exception("User simulator LLM invocation failed case_id=%s", case.id)
            raise UserSimulatorError(
                "llm_invocation_failed",
                "User simulator failed to generate a reply.",
            ) from exc

        answer = self._extract_text_response(response)
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

    def _get_case(self, case_id: int) -> NonDeterministicRegressionCase:
        """Return one case or raise a typed error."""

        case = self._case_index.get(case_id)
        if case is None:
            raise UserSimulatorError(
                "unknown_case_id",
                f"Unknown non-deterministic regression case id: {case_id}.",
            )
        return case

    def _build_reply_messages(
        self,
        case: NonDeterministicRegressionCase,
        agent_message: str,
        conversation: list[ConversationTurn],
    ) -> list[object]:
        """Build the text-only LLM prompt for one follow-up reply."""

        behavioral_traits = "\n".join(
            f"- {trait}" for trait in case.communication_style.behavioral_traits
        )
        filter_criteria = "\n".join(f"- {criterion}" for criterion in case.filter_criteria)
        axes = "\n".join(f"- {axis}" for axis in case.axes)
        conversation_so_far = _format_conversation(conversation)

        system_message = SystemMessage(
            content=(
                "You are a user simulator. Your job is to write the next message the "
                "user would send in this conversation.\n\n"
                "A user simulator answers like the user, not like the agent. Reply to "
                "the agent's latest message as a real user continuing a practical back-"
                "and-forth chat, without customer-service politeness. Stay "
                "consistent with the user's background and goals, but do not restate "
                "the whole profile unless the latest message makes that necessary.\n\n"
                "Priority rules:\n"
                "1. Answer the latest agent message directly.\n"
                "2. Keep visible conversation text separate from private case data: "
                "the initial prompt and transcript show what the user has said, while "
                "case data describes the user's intended background and preferences.\n"
                "3. Keep every reply strictly aligned with the initial prompt and the "
                "provided case data.\n"
                "4. Do not contradict, weaken, or drift away from the initial prompt's "
                "stated goals, regions, constraints, or priorities unless the agent's "
                "latest message gives a valid reason to narrow in on one part of them.\n"
                "5. Do not invent unsupported facts. If a detail is not established by "
                "the visible conversation or private case data, avoid guessing.\n"
                "6. If the agent asks for a detail that is already stated or strongly "
                "implied by the visible conversation or private case data, answer with "
                "only the grounded fact that addresses it.\n"
                "7. If the agent asks a forced-choice question and neither option is "
                "fully supported by the case data, do not pick one just to be helpful. "
                "Answer only what is grounded and briefly note what is not established.\n"
                "8. For redundant, low-signal, or over-specific questions, reply the "
                "way a real user would: short, direct, and grounded in what was already "
                "said. It is fine to point back to the original message.\n"
                "9. Be especially careful with employment status, visa mechanics, "
                "timeline, compensation, employer names, and relocation facts. Do not "
                "add specifics unless they are explicitly established.\n"
                "10. Use the initial prompt and the person's communication style to "
                "preserve the user's tone of voice, phrasing style, and level of "
                "directness.\n"
                "11. Use the person's communication style only to shape tone and "
                "structure.\n"
                "12. Do not repeat or summarize all prior context unless needed.\n"
                "13. Do not turn the reply into a polished memo, cover letter, or "
                "scripted interview answer.\n"
                "14. Prefer natural, utilitarian user phrasing over polished, "
                "affirming, or overly courteous language.\n"
                "15. Do not sound like a support agent, coach, or survey "
                "respondent.\n"
                "16. Focus on the user's practical need, not on the quality, effort, "
                "or helpfulness of the agent's message.\n"
                "17. Do not add meta-commentary about the conversation unless the "
                "latest message clearly calls for it.\n\n"
                "Reply rules:\n"
                "- Output plain text only.\n"
                "- Write one user message only.\n"
                "- Start with the substantive answer immediately, with minimal social "
                "framing.\n"
                "- Keep it short to medium unless the agent asks for detail.\n"
                "- Do not use bullet points, headings, numbered lists, or formal "
                "closers unless explicitly asked.\n"
                "- Do not invent a new career goal, region, or profile detail unless "
                "directly implied by the case.\n"
                "- If the question was already answered by the initial prompt, prefer a "
                "brief correction or restatement over a fresh explanation.\n"
                "- Respond to the substance of the agent's message, not to its tone, "
                "effort, or usefulness.\n"
                "- Treat the initial prompt as the strongest signal for how the user "
                "sounds and what the user cares about.\n"
                "- Do not reuse or echo the initial prompt's wording unless directly "
                "relevant.\n"
                "- Use the case communication style to shape formality and directness, "
                "but keep the reply human and slightly imperfect rather than polished."
            )
        )
        human_message = HumanMessage(
            content=(
                "You need to simulate the user's reply to this latest agent message:\n\n"
                f"{agent_message}\n\n"
                "Private case data that describes the simulated user's intent, not "
                "necessarily prior wording:\n"
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
        )
        return [system_message, human_message]

    @staticmethod
    def _extract_text_response(response: object) -> str:
        """Normalize one model response into plain text."""

        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif hasattr(item, "text") and isinstance(item.text, str):
                    text_parts.append(item.text)
            return "\n".join(part.strip() for part in text_parts if part.strip()).strip()
        return str(content).strip()


def _format_conversation(conversation: list[ConversationTurn]) -> str:
    """Return a compact transcript block for simulator grounding."""

    if not conversation:
        return "No prior turns were provided."

    lines: list[str] = []
    for turn in conversation:
        speaker = turn.speaker.strip().lower()
        label = "User" if speaker == "user" else "Agent"
        message = " ".join(turn.message.split())
        if message:
            lines.append(f"{label}: {message}")
    return "\n".join(lines) or "No prior turns were provided."


@lru_cache(maxsize=1)
def create_default_user_simulator() -> UserSimulator:
    """Return a default stateless user simulator instance."""

    return UserSimulator()
