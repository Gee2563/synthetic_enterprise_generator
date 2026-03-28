from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from synthetic_enterprise.domain import EnterpriseGraph, EventScenarioResolver
from synthetic_enterprise.generation.context import GeneratorContext

AttendanceRSVPStatus = Literal["accepted", "tentative", "declined"]
AttendanceFinalStatus = Literal["attended", "no_show", "declined"]


@dataclass(frozen=True, slots=True)
class AttendanceParticipantState:
    contact_id: str
    invited_at: datetime
    rsvp_requested_at: datetime
    reminder_sent_at: datetime
    rsvp_status: AttendanceRSVPStatus
    final_status: AttendanceFinalStatus


@dataclass(frozen=True, slots=True)
class EventAttendanceLifecycle:
    event_id: str
    invite_sent_at: datetime
    rsvp_requested_at: datetime
    reminder_sent_at: datetime
    internal_coordination_at: datetime
    event_started_at: datetime
    event_ended_at: datetime
    post_event_follow_up_at: datetime
    participants: tuple[AttendanceParticipantState, ...]

    @property
    def attended_contact_ids(self) -> tuple[str, ...]:
        return tuple(
            participant.contact_id
            for participant in self.participants
            if participant.final_status == "attended"
        )

    @property
    def no_show_contact_ids(self) -> tuple[str, ...]:
        return tuple(
            participant.contact_id
            for participant in self.participants
            if participant.final_status == "no_show"
        )


@dataclass(slots=True)
class EventAttendanceBuilder:
    context: GeneratorContext
    enterprise: EnterpriseGraph

    def build(self, *, event_id: str) -> EventAttendanceLifecycle:
        scenario = EventScenarioResolver(self.enterprise).for_event(event_id)
        event = scenario.event
        invite_sent_at = event.starts_at - timedelta(days=7)
        rsvp_requested_at = invite_sent_at + timedelta(days=1)
        reminder_sent_at = event.starts_at - timedelta(days=1)
        internal_coordination_at = event.starts_at - timedelta(hours=3)
        post_event_follow_up_at = event.ends_at + timedelta(hours=2)

        participants: list[AttendanceParticipantState] = []
        for index, contact_id in enumerate(event.attendee_contact_ids):
            if index == 0:
                rsvp_status: AttendanceRSVPStatus = "accepted"
                final_status: AttendanceFinalStatus = "attended"
            elif index == 1:
                rsvp_status = "accepted"
                final_status = "no_show"
            else:
                rsvp_status = "declined"
                final_status = "declined"

            participants.append(
                AttendanceParticipantState(
                    contact_id=contact_id,
                    invited_at=invite_sent_at,
                    rsvp_requested_at=rsvp_requested_at,
                    reminder_sent_at=reminder_sent_at,
                    rsvp_status=rsvp_status,
                    final_status=final_status,
                )
            )

        return EventAttendanceLifecycle(
            event_id=event.id,
            invite_sent_at=invite_sent_at,
            rsvp_requested_at=rsvp_requested_at,
            reminder_sent_at=reminder_sent_at,
            internal_coordination_at=internal_coordination_at,
            event_started_at=event.starts_at,
            event_ended_at=event.ends_at,
            post_event_follow_up_at=post_event_follow_up_at,
            participants=tuple(participants),
        )
