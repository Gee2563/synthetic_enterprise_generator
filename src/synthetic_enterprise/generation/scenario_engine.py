from __future__ import annotations

from synthetic_enterprise.contracts.company_state import CompanyState
from synthetic_enterprise.contracts.scenario import Scenario


class ScenarioEngine:
    """Placeholder scenario derivation component."""

    def build(self, company_state: CompanyState) -> list[Scenario]:
        return [
            Scenario(
                scenario_id=f"scenario-{company_state.seed}",
                kind="placeholder",
                company_id=company_state.company_id,
            )
        ]
