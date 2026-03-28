from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from synthetic_enterprise.contracts.company_state import CompanyState
from synthetic_enterprise.contracts.scenario import (
    Scenario,
    ScenarioCategoryGrounding,
    ScenarioCommunicationTrigger,
    ScenarioEntityStateTransition,
    ScenarioEntityStateType,
    ScenarioFamily,
    ScenarioLifecycleStage,
    ScenarioStateTransition,
    ScenarioSystem,
)
from synthetic_enterprise.domain import Contact, CustomerAccount, EnterpriseGraph
from synthetic_enterprise.domain.base import deterministic_timestamp, stable_entity_id
from synthetic_enterprise.generation.account_profiles import AccountBehaviorResolver
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.grounding import (
    LabelProvenance,
    ProvenanceObjectType,
)
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory


@dataclass(frozen=True, slots=True)
class _ScenarioBlueprint:
    family: ScenarioFamily
    expected_labels: tuple[CommunicationCategory, ...]
    stages: tuple[ScenarioLifecycleStage, ...]
    surface_systems: tuple[ScenarioSystem, ...]


SCENARIO_BLUEPRINTS: tuple[_ScenarioBlueprint, ...] = (
    _ScenarioBlueprint(
        family=ScenarioFamily.PRE_SALES_DISCOVERY,
        expected_labels=(
            CommunicationCategory.BUYING_SIGNAL,
            CommunicationCategory.FOLLOW_UP,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.COMPLETED,
        ),
        surface_systems=("email", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.PROCUREMENT_DELAY,
        expected_labels=(
            CommunicationCategory.BLOCKER,
            CommunicationCategory.FOLLOW_UP,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.STALLED,
        ),
        surface_systems=("email", "slack", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.PILOT_SUCCESS,
        expected_labels=(
            CommunicationCategory.BUYING_SIGNAL,
            CommunicationCategory.EVENT_ATTENDANCE,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.COMPLETED,
        ),
        surface_systems=("email", "teams", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.ROLLOUT_RISK,
        expected_labels=(
            CommunicationCategory.BLOCKER,
            CommunicationCategory.CHURN_RISK,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.STALLED,
        ),
        surface_systems=("slack", "teams", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.SUPPORT_ESCALATION,
        expected_labels=(
            CommunicationCategory.PAIN_POINT,
            CommunicationCategory.ESCALATION,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.STALLED,
            ScenarioLifecycleStage.REOPENED,
        ),
        surface_systems=("email", "slack", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.FEATURE_GAP_REVIEW,
        expected_labels=(
            CommunicationCategory.FEATURE_REQUEST,
            CommunicationCategory.DECISION_MAKER_SIGNAL,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.COMPLETED,
        ),
        surface_systems=("email", "teams", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.RENEWAL_RISK,
        expected_labels=(
            CommunicationCategory.CHURN_RISK,
            CommunicationCategory.FOLLOW_UP,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.STALLED,
        ),
        surface_systems=("email", "slack", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.EXECUTIVE_SPONSOR_REVIEW,
        expected_labels=(
            CommunicationCategory.DECISION_MAKER_SIGNAL,
            CommunicationCategory.FOLLOW_UP,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.COMPLETED,
        ),
        surface_systems=("email", "teams", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.EVENT_INVITE_TO_ATTENDANCE,
        expected_labels=(
            CommunicationCategory.EVENT_ATTENDANCE,
            CommunicationCategory.FOLLOW_UP,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.COMPLETED,
        ),
        surface_systems=("email", "teams", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.CHAMPION_DEPARTURE,
        expected_labels=(
            CommunicationCategory.CHURN_RISK,
            CommunicationCategory.BLOCKER,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.STALLED,
            ScenarioLifecycleStage.REOPENED,
        ),
        surface_systems=("slack", "teams", "salesforce"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.STAKEHOLDER_EXPANSION,
        expected_labels=(
            CommunicationCategory.DECISION_MAKER_SIGNAL,
            CommunicationCategory.BUYING_SIGNAL,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.COMPLETED,
        ),
        surface_systems=("email", "slack", "teams"),
    ),
    _ScenarioBlueprint(
        family=ScenarioFamily.CONTRACT_REDLINE_DELAY,
        expected_labels=(
            CommunicationCategory.BLOCKER,
            CommunicationCategory.FOLLOW_UP,
        ),
        stages=(
            ScenarioLifecycleStage.IDENTIFIED,
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.STALLED,
        ),
        surface_systems=("email", "salesforce"),
    ),
)


@dataclass(slots=True)
class ScenarioEngine:
    """Derive temporal business arcs from deterministic enterprise state."""

    context: GeneratorContext | None = None

    def build(self, company_state: CompanyState) -> list[Scenario]:
        return self.build_legacy(
            company_id=company_state.company_id,
            company_name=company_state.company_name,
            seed=company_state.seed,
        )

    def build_legacy(
        self,
        *,
        company_id: str,
        company_name: str,
        seed: int,
    ) -> list[Scenario]:
        scenario_id = stable_entity_id("scenario", seed, f"legacy:{company_id}")
        transition = ScenarioStateTransition(
            transition_id=stable_entity_id(
                "scenario_transition",
                seed,
                f"{scenario_id}:identified",
            ),
            stage=ScenarioLifecycleStage.IDENTIFIED,
            occurred_at=deterministic_timestamp(seed, f"{scenario_id}:identified"),
            summary=f"Legacy placeholder scenario for {company_name}.",
            state_changes={"legacy_mode": "initialized"},
        )
        return [
            Scenario(
                scenario_id=scenario_id,
                kind="placeholder",
                company_id=company_id,
                transitions=(transition,),
                current_stage=ScenarioLifecycleStage.IDENTIFIED,
                expected_labels=(),
            )
        ]

    def build_for_enterprise(self, enterprise: EnterpriseGraph) -> list[Scenario]:
        if self.context is None:
            raise ValueError("ScenarioEngine.build_for_enterprise requires a GeneratorContext")

        context = self.context
        company = enterprise.companies[0]
        accounts = sorted(enterprise.customer_accounts, key=lambda account: account.id)
        all_employee_ids = tuple(sorted(employee.id for employee in enterprise.employees))
        opportunities_by_account = {
            opportunity.account_id: opportunity
            for opportunity in enterprise.opportunities
        }
        events_by_account = {
            event.account_id: event
            for event in enterprise.events
            if event.account_id is not None
        }
        tickets_by_account = {
            ticket.account_id: ticket
            for ticket in enterprise.ticket_issues
        }
        contacts_by_account: dict[str, list[Contact]] = {}
        for contact in enterprise.contacts:
            contacts_by_account.setdefault(contact.account_id, []).append(contact)
        employees_by_id = {employee.id: employee for employee in enterprise.employees}
        behavior_resolver = AccountBehaviorResolver(
            context=context,
            enterprise=enterprise,
        )

        def build_scenario(
            *,
            blueprint: _ScenarioBlueprint,
            account: CustomerAccount,
        ) -> Scenario:
            contacts = tuple(
                sorted(
                    contacts_by_account.get(account.id, []),
                    key=lambda contact: contact.id,
                )
            )
            opportunity = opportunities_by_account.get(account.id)
            event = events_by_account.get(account.id)
            ticket = tickets_by_account.get(account.id)
            owner = employees_by_id[account.owner_employee_id]
            participant_employees = [owner]
            if event is not None:
                participant_employees.append(employees_by_id[event.organizer_employee_id])
            if ticket is not None and ticket.owner_employee_id is not None:
                participant_employees.append(employees_by_id[ticket.owner_employee_id])
            employee_ids = tuple(
                sorted({employee.id for employee in participant_employees})
            )
            contact_ids = tuple(contact.id for contact in contacts[:2])
            fallback_contact_id = contact_ids[0] if contact_ids else None
            scenario_seed = context.derive_seed(
                f"scenario:{blueprint.family.value}:{account.id}"
            )
            alternate_owner_id = self._alternate_employee_id(
                employee_ids=all_employee_ids,
                current_owner_id=owner.id,
                scenario_seed=scenario_seed,
            )
            scenario_id = stable_entity_id(
                "scenario",
                scenario_seed,
                f"{blueprint.family.value}:{company.id}:{account.id}",
            )
            transitions = self._build_transitions(
                scenario_id=scenario_id,
                scenario_seed=scenario_seed,
                blueprint=blueprint,
                account_name=account.name,
                account_id=account.id,
                opportunity_id=opportunity.id if opportunity is not None else None,
                event_id=event.id if event is not None else None,
                ticket_id=ticket.id if ticket is not None else None,
                primary_contact_id=fallback_contact_id,
                alternate_owner_id=alternate_owner_id,
                secondary_contact_id=(
                    contact_ids[1] if len(contact_ids) > 1 else fallback_contact_id
                ),
                current_owner_id=owner.id,
            )

            return Scenario(
                scenario_id=scenario_id,
                kind=blueprint.family,
                company_id=company.id,
                account_ids=(account.id,),
                employee_ids=employee_ids,
                contact_ids=contact_ids,
                opportunity_ids=((opportunity.id,) if opportunity is not None else ()),
                event_ids=((event.id,) if event is not None else ()),
                ticket_ids=((ticket.id,) if ticket is not None else ()),
                surface_systems=blueprint.surface_systems,
                transitions=tuple(transitions),
                current_stage=blueprint.stages[-1],
                expected_labels=blueprint.expected_labels,
            )

        scenarios: list[Scenario] = []
        primary_family_by_account_id: dict[str, ScenarioFamily] = {}
        for index, blueprint in enumerate(SCENARIO_BLUEPRINTS):
            if index >= len(accounts):
                break

            account = accounts[index]
            scenarios.append(build_scenario(blueprint=blueprint, account=account))
            primary_family_by_account_id[account.id] = blueprint.family

        blueprints_by_family = {
            blueprint.family: blueprint
            for blueprint in SCENARIO_BLUEPRINTS
        }
        for account in accounts:
            profile = behavior_resolver.profile_for_account(account.id)
            preferred_family = profile.scheduled_family
            if primary_family_by_account_id.get(account.id) == preferred_family:
                continue
            scenarios.append(
                build_scenario(
                    blueprint=blueprints_by_family[preferred_family],
                    account=account,
                )
            )

        return scenarios

    def _build_transitions(
        self,
        *,
        scenario_id: str,
        scenario_seed: int,
        blueprint: _ScenarioBlueprint,
        account_name: str,
        account_id: str,
        opportunity_id: str | None,
        event_id: str | None,
        ticket_id: str | None,
        primary_contact_id: str | None,
        alternate_owner_id: str,
        secondary_contact_id: str | None,
        current_owner_id: str,
    ) -> list[ScenarioStateTransition]:
        transitions: list[ScenarioStateTransition] = []
        previous_at = deterministic_timestamp(
            scenario_seed,
            f"{scenario_id}:identified:base",
        )

        for index, stage in enumerate(blueprint.stages):
            if index == 0:
                occurred_at = previous_at
            else:
                occurred_at = deterministic_timestamp(
                    scenario_seed,
                    f"{scenario_id}:{stage.value}:{index}",
                    floor=previous_at + timedelta(minutes=1),
                )
            previous_at = occurred_at
            entity_state_transitions = self._entity_state_transitions(
                blueprint=blueprint,
                stage=stage,
                account_id=account_id,
                opportunity_id=opportunity_id,
                event_id=event_id,
                ticket_id=ticket_id,
                    primary_contact_id=primary_contact_id,
                    secondary_contact_id=secondary_contact_id,
                    alternate_owner_id=alternate_owner_id,
                    current_owner_id=current_owner_id,
                )
            transitions.append(
                ScenarioStateTransition(
                    transition_id=stable_entity_id(
                        "scenario_transition",
                        scenario_seed,
                        f"{scenario_id}:{stage.value}:{index}",
                    ),
                    stage=stage,
                    occurred_at=occurred_at,
                    summary=(
                        f"{blueprint.family.value} for {account_name} moved to "
                        f"{stage.value.replace('_', ' ')}."
                    ),
                    state_changes=self._state_changes(
                        blueprint=blueprint,
                        stage=stage,
                        opportunity_id=opportunity_id,
                        event_id=event_id,
                        ticket_id=ticket_id,
                    ),
                    entity_state_transitions=tuple(entity_state_transitions),
                    communication_triggers=self._communication_triggers(
                        blueprint=blueprint,
                        stage=stage,
                        entity_state_transitions=tuple(entity_state_transitions),
                    ),
                    category_groundings=self._category_groundings(
                        blueprint=blueprint,
                        stage=stage,
                        account_id=account_id,
                        opportunity_id=opportunity_id,
                        event_id=event_id,
                        ticket_id=ticket_id,
                        primary_contact_id=primary_contact_id,
                    ),
                )
            )

        return transitions

    def _alternate_employee_id(
        self,
        *,
        employee_ids: tuple[str, ...],
        current_owner_id: str,
        scenario_seed: int,
    ) -> str:
        alternatives = [
            employee_id
            for employee_id in employee_ids
            if employee_id != current_owner_id
        ]
        if not alternatives:
            return current_owner_id
        return alternatives[scenario_seed % len(alternatives)]

    def _entity_state_transitions(
        self,
        *,
        blueprint: _ScenarioBlueprint,
        stage: ScenarioLifecycleStage,
        account_id: str,
        opportunity_id: str | None,
        event_id: str | None,
        ticket_id: str | None,
        primary_contact_id: str | None,
        secondary_contact_id: str | None,
        alternate_owner_id: str,
        current_owner_id: str,
    ) -> list[ScenarioEntityStateTransition]:
        transitions: list[ScenarioEntityStateTransition] = []

        def add(
            transition_type: ScenarioEntityStateType,
            entity_id: str | None,
            from_state: str | None,
            to_state: str,
            *,
            actor_id: str | None = None,
        ) -> None:
            if entity_id is None:
                return
            transitions.append(
                ScenarioEntityStateTransition(
                    transition_type=transition_type,
                    entity_id=entity_id,
                    from_state=from_state,
                    to_state=to_state,
                    actor_id=actor_id,
                )
            )

        family = blueprint.family

        if family == ScenarioFamily.PRE_SALES_DISCOVERY:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(ScenarioEntityStateType.OPPORTUNITY_STAGE, opportunity_id, None, "discovery")
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    None,
                    "engaged",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    opportunity_id or account_id,
                    None,
                    "pending",
                )
            elif stage == ScenarioLifecycleStage.COMPLETED:
                add(
                    ScenarioEntityStateType.OPPORTUNITY_STAGE,
                    opportunity_id,
                    "discovery",
                    "evaluation",
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    opportunity_id or account_id,
                    "pending",
                    "completed",
                )
        elif family == ScenarioFamily.PROCUREMENT_DELAY:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(
                    ScenarioEntityStateType.OPPORTUNITY_STAGE,
                    opportunity_id,
                    "evaluation",
                    "proposal",
                )
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    opportunity_id or account_id,
                    None,
                    "pending",
                )
            elif stage == ScenarioLifecycleStage.STALLED:
                add(
                    ScenarioEntityStateType.OWNERSHIP,
                    opportunity_id or account_id,
                    current_owner_id,
                    alternate_owner_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    opportunity_id or account_id,
                    "pending",
                    "slipped",
                )
        elif family == ScenarioFamily.PILOT_SUCCESS:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(
                    ScenarioEntityStateType.EVENT_PARTICIPATION,
                    event_id,
                    None,
                    "invited",
                    actor_id=primary_contact_id,
                )
                add(ScenarioEntityStateType.ACCOUNT_HEALTH, account_id, None, "healthy")
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(
                    ScenarioEntityStateType.EVENT_PARTICIPATION,
                    event_id,
                    "invited",
                    "accepted",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    event_id or account_id,
                    None,
                    "pending",
                )
            elif stage == ScenarioLifecycleStage.COMPLETED:
                add(
                    ScenarioEntityStateType.EVENT_PARTICIPATION,
                    event_id,
                    "accepted",
                    "attended",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    event_id or account_id,
                    "pending",
                    "completed",
                )
        elif family == ScenarioFamily.ROLLOUT_RISK:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(ScenarioEntityStateType.TICKET_SEVERITY, ticket_id, None, "medium")
                add(ScenarioEntityStateType.ACCOUNT_HEALTH, account_id, None, "watchlist")
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(ScenarioEntityStateType.TICKET_SEVERITY, ticket_id, "medium", "high")
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    ticket_id or account_id,
                    None,
                    "pending",
                )
            elif stage == ScenarioLifecycleStage.STALLED:
                add(ScenarioEntityStateType.TICKET_SEVERITY, ticket_id, "high", "critical")
                add(ScenarioEntityStateType.ACCOUNT_HEALTH, account_id, "watchlist", "at_risk")
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    ticket_id or account_id,
                    "pending",
                    "slipped",
                )
        elif family == ScenarioFamily.SUPPORT_ESCALATION:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(ScenarioEntityStateType.TICKET_SEVERITY, ticket_id, None, "high")
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    ticket_id or account_id,
                    None,
                    "pending",
                )
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(ScenarioEntityStateType.TICKET_SEVERITY, ticket_id, "high", "critical")
            elif stage == ScenarioLifecycleStage.STALLED:
                add(
                    ScenarioEntityStateType.OWNERSHIP,
                    ticket_id or account_id,
                    current_owner_id,
                    alternate_owner_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    ticket_id or account_id,
                    "pending",
                    "slipped",
                )
            elif stage == ScenarioLifecycleStage.REOPENED:
                add(ScenarioEntityStateType.TICKET_SEVERITY, ticket_id, "critical", "high")
        elif family == ScenarioFamily.FEATURE_GAP_REVIEW:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    None,
                    "engaged",
                    actor_id=primary_contact_id,
                )
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    opportunity_id or account_id,
                    None,
                    "pending",
                )
            elif stage == ScenarioLifecycleStage.COMPLETED:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    "engaged",
                    "expanding",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    opportunity_id or account_id,
                    "pending",
                    "completed",
                )
        elif family == ScenarioFamily.RENEWAL_RISK:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(ScenarioEntityStateType.ACCOUNT_HEALTH, account_id, None, "watchlist")
                add(
                    ScenarioEntityStateType.OPPORTUNITY_STAGE,
                    opportunity_id,
                    "discovery",
                    "evaluation",
                )
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    None,
                    "neutral",
                    actor_id=primary_contact_id,
                )
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(ScenarioEntityStateType.ACCOUNT_HEALTH, account_id, "watchlist", "at_risk")
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    "neutral",
                    "disengaged",
                    actor_id=primary_contact_id,
                )
                add(ScenarioEntityStateType.FOLLOW_UP_STATUS, account_id, None, "pending")
            elif stage == ScenarioLifecycleStage.STALLED:
                add(
                    ScenarioEntityStateType.OPPORTUNITY_STAGE,
                    opportunity_id,
                    "evaluation",
                    "proposal",
                )
                add(ScenarioEntityStateType.FOLLOW_UP_STATUS, account_id, "pending", "slipped")
        elif family == ScenarioFamily.EXECUTIVE_SPONSOR_REVIEW:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    None,
                    "engaged",
                    actor_id=primary_contact_id,
                )
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    "engaged",
                    "expanding",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    event_id or account_id,
                    None,
                    "pending",
                )
            elif stage == ScenarioLifecycleStage.COMPLETED:
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    event_id or account_id,
                    "pending",
                    "completed",
                )
        elif family == ScenarioFamily.EVENT_INVITE_TO_ATTENDANCE:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(
                    ScenarioEntityStateType.EVENT_PARTICIPATION,
                    event_id,
                    None,
                    "invited",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    event_id or account_id,
                    None,
                    "pending",
                )
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(
                    ScenarioEntityStateType.EVENT_PARTICIPATION,
                    event_id,
                    "invited",
                    "accepted",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.EVENT_PARTICIPATION,
                    event_id,
                    "accepted",
                    "tentative",
                    actor_id=primary_contact_id,
                )
            elif stage == ScenarioLifecycleStage.COMPLETED:
                add(
                    ScenarioEntityStateType.EVENT_PARTICIPATION,
                    event_id,
                    "tentative",
                    "attended",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    event_id or account_id,
                    "pending",
                    "completed",
                )
        elif family == ScenarioFamily.CHAMPION_DEPARTURE:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    None,
                    "engaged",
                    actor_id=primary_contact_id,
                )
                add(ScenarioEntityStateType.ACCOUNT_HEALTH, account_id, None, "healthy")
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    "engaged",
                    "disengaged",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.OWNERSHIP,
                    account_id,
                    current_owner_id,
                    alternate_owner_id,
                )
            elif stage == ScenarioLifecycleStage.STALLED:
                add(ScenarioEntityStateType.ACCOUNT_HEALTH, account_id, "healthy", "watchlist")
                add(ScenarioEntityStateType.FOLLOW_UP_STATUS, account_id, None, "pending")
            elif stage == ScenarioLifecycleStage.REOPENED:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    "disengaged",
                    "engaged",
                    actor_id=primary_contact_id,
                )
                add(ScenarioEntityStateType.FOLLOW_UP_STATUS, account_id, "pending", "completed")
        elif family == ScenarioFamily.STAKEHOLDER_EXPANSION:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    primary_contact_id,
                    None,
                    "engaged",
                    actor_id=primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.OPPORTUNITY_STAGE,
                    opportunity_id,
                    None,
                    "discovery",
                )
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(
                    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
                    secondary_contact_id or primary_contact_id,
                    "engaged",
                    "expanding",
                    actor_id=secondary_contact_id or primary_contact_id,
                )
                add(
                    ScenarioEntityStateType.OPPORTUNITY_STAGE,
                    opportunity_id,
                    "discovery",
                    "evaluation",
                )
            elif stage == ScenarioLifecycleStage.COMPLETED:
                add(
                    ScenarioEntityStateType.OPPORTUNITY_STAGE,
                    opportunity_id,
                    "evaluation",
                    "proposal",
                )
        elif family == ScenarioFamily.CONTRACT_REDLINE_DELAY:
            if stage == ScenarioLifecycleStage.IDENTIFIED:
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    opportunity_id or account_id,
                    None,
                    "pending",
                )
            elif stage == ScenarioLifecycleStage.ACTIVE:
                add(
                    ScenarioEntityStateType.OPPORTUNITY_STAGE,
                    opportunity_id,
                    "evaluation",
                    "proposal",
                )
            elif stage == ScenarioLifecycleStage.STALLED:
                add(
                    ScenarioEntityStateType.OWNERSHIP,
                    opportunity_id or account_id,
                    current_owner_id,
                    alternate_owner_id,
                )
                add(
                    ScenarioEntityStateType.FOLLOW_UP_STATUS,
                    opportunity_id or account_id,
                    "pending",
                    "slipped",
                )

        return transitions

    def _communication_triggers(
        self,
        *,
        blueprint: _ScenarioBlueprint,
        stage: ScenarioLifecycleStage,
        entity_state_transitions: tuple[ScenarioEntityStateTransition, ...],
    ) -> tuple[ScenarioCommunicationTrigger, ...]:
        if not entity_state_transitions:
            return ()

        return tuple(
            ScenarioCommunicationTrigger(
                category=category,
                reason=(
                    f"{blueprint.family.value} entered {stage.value} after "
                    f"{entity_state_transitions[0].transition_type.value} changed."
                ),
            )
            for category in blueprint.expected_labels
        )

    def _state_changes(
        self,
        *,
        blueprint: _ScenarioBlueprint,
        stage: ScenarioLifecycleStage,
        opportunity_id: str | None,
        event_id: str | None,
        ticket_id: str | None,
    ) -> dict[str, str]:
        state_changes = {
            "scenario_family": blueprint.family.value,
            "scenario_stage": stage.value,
        }
        if opportunity_id is not None:
            state_changes["linked_opportunity"] = opportunity_id
        if event_id is not None:
            state_changes["linked_event"] = event_id
        if ticket_id is not None:
            state_changes["linked_ticket"] = ticket_id
        return state_changes

    def _category_groundings(
        self,
        *,
        blueprint: _ScenarioBlueprint,
        stage: ScenarioLifecycleStage,
        account_id: str,
        opportunity_id: str | None,
        event_id: str | None,
        ticket_id: str | None,
        primary_contact_id: str | None,
    ) -> tuple[ScenarioCategoryGrounding, ...]:
        if stage not in {
            ScenarioLifecycleStage.ACTIVE,
            ScenarioLifecycleStage.STALLED,
            ScenarioLifecycleStage.REOPENED,
            ScenarioLifecycleStage.COMPLETED,
        }:
            return ()

        return tuple(
            ScenarioCategoryGrounding(
                category=category,
                provenance=self._provenance_for_category(
                    category=category,
                    account_id=account_id,
                    opportunity_id=opportunity_id,
                    event_id=event_id,
                    ticket_id=ticket_id,
                    primary_contact_id=primary_contact_id,
                ),
            )
            for category in blueprint.expected_labels
        )

    def _provenance_for_category(
        self,
        *,
        category: CommunicationCategory,
        account_id: str,
        opportunity_id: str | None,
        event_id: str | None,
        ticket_id: str | None,
        primary_contact_id: str | None,
    ) -> LabelProvenance:
        if category == CommunicationCategory.BUYING_SIGNAL and opportunity_id is not None:
            return LabelProvenance(
                object_type=ProvenanceObjectType.OPPORTUNITY,
                object_id=opportunity_id,
                explanation="Commercial progress is visible in the scenario state.",
            )
        if category == CommunicationCategory.EVENT_ATTENDANCE and event_id is not None:
            return LabelProvenance(
                object_type=ProvenanceObjectType.EVENT,
                object_id=event_id,
                explanation="Attendance progression is captured in the scenario lifecycle.",
            )
        if category in {
            CommunicationCategory.PAIN_POINT,
            CommunicationCategory.COMPLAINT,
            CommunicationCategory.BLOCKER,
            CommunicationCategory.ESCALATION,
            CommunicationCategory.FEATURE_REQUEST,
        } and ticket_id is not None:
            return LabelProvenance(
                object_type=ProvenanceObjectType.TICKET,
                object_id=ticket_id,
                explanation="Open issue state is driving the scenario transition.",
            )
        if (
            category == CommunicationCategory.DECISION_MAKER_SIGNAL
            and primary_contact_id is not None
        ):
            return LabelProvenance(
                object_type=ProvenanceObjectType.CONTACT,
                object_id=primary_contact_id,
                explanation="Contact involvement in the scenario reflects stakeholder authority.",
            )
        if category == CommunicationCategory.FOLLOW_UP and event_id is not None:
            return LabelProvenance(
                object_type=ProvenanceObjectType.EVENT,
                object_id=event_id,
                explanation="Next action timing is anchored to a scheduled customer event.",
            )
        if category == CommunicationCategory.CHURN_RISK:
            target_type = (
                ProvenanceObjectType.TICKET
                if ticket_id is not None
                else ProvenanceObjectType.ACCOUNT
            )
            target_id = ticket_id if ticket_id is not None else account_id
            return LabelProvenance(
                object_type=target_type,
                object_id=target_id,
                explanation="Scenario state shows unresolved risk against the account timeline.",
            )

        return LabelProvenance(
            object_type=ProvenanceObjectType.ACCOUNT,
            object_id=account_id,
            explanation="Scenario state is linked directly to the customer account.",
        )
