"""FSM states for multi-step flows."""

from aiogram.fsm.state import State, StatesGroup


class TutorFlow(StatesGroup):
    """Tutor dialogue (CLAUDE.md §7 Phase 1).

    waiting_for_topic -> the bot asked what to learn, nothing started yet
    in_lesson         -> explanation sent, waiting for the student's answer
    """

    waiting_for_topic = State()
    in_lesson = State()
