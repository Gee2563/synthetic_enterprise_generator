"""Core enterprise domain schemas and aggregate relationship validation."""

from synthetic_enterprise.domain.accounts import CustomerAccount
from synthetic_enterprise.domain.activities import CRMActivity, MessageEnvelope
from synthetic_enterprise.domain.company import Company, Department
from synthetic_enterprise.domain.enterprise import EnterpriseGraph
from synthetic_enterprise.domain.events import Campaign, Event
from synthetic_enterprise.domain.issues import TicketIssue
from synthetic_enterprise.domain.opportunities import Opportunity
from synthetic_enterprise.domain.people import Contact, Employee
from synthetic_enterprise.domain.product import Product
from synthetic_enterprise.domain.scenarios import EventScenarioContext, EventScenarioResolver

__all__ = [
    "CRMActivity",
    "Campaign",
    "Company",
    "Contact",
    "CustomerAccount",
    "Department",
    "Employee",
    "EnterpriseGraph",
    "EventScenarioContext",
    "EventScenarioResolver",
    "Event",
    "MessageEnvelope",
    "Opportunity",
    "Product",
    "TicketIssue",
]
