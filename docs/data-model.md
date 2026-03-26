# Data Model

## Enterprise Graph

The central domain object is `EnterpriseGraph`. It collects entities and validates relationships across the simulated company.

Core entities:

- `Company`: top-level enterprise identity and timezone
- `Department`: organizational structure and leadership
- `Employee`: internal actors, roles, and reporting lines
- `CustomerAccount`: customer accounts owned by employees
- `Contact`: external people tied to accounts
- `Opportunity`: commercial motion linked to accounts and contacts
- `Event`: meetings, reviews, or attendance-bearing activities
- `Product`: products relevant to deals and issues
- `TicketIssue`: support problems and blockers
- `Campaign`: campaign-level CRM grouping
- `MessageEnvelope`: message metadata in the domain layer
- `CRMActivity`: CRM-side interaction history

## Relationship Rules

Examples of explicit relationship checks already enforced by the models:

- every employee belongs to a valid department
- department leaders reference valid employees
- account owners reference valid employees in the same company
- contacts reference valid customer accounts
- opportunities reference valid accounts, contacts, owners, and products
- events reference valid accounts, organizers, and attendee contacts
- ticket issues reference valid accounts, contacts, and optional products
- campaigns reference valid accounts, products, and owners

The graph rejects broken references and globally duplicate ids.

## Stable Identity

Every major entity and row has a stable deterministic id. Examples:

- `company_<seed-derived>`
- `employee_<seed-derived>`
- `account_<seed-derived>`
- `event_<seed-derived>`
- `ticket_issue_<seed-derived>`
- `email_<seed-derived>`
- `slack_message_<seed-derived>`
- `teams_message_<seed-derived>`
- `sf_record_<seed-derived>`

These ids are not random UUIDs. They are derived from the root seed and a namespace.

## Time Model

All domain entities and source rows use timezone-aware datetimes.

- domain entities carry `created_at` and `updated_at`
- source rows carry channel-specific timestamps
- generator config bounds the legal date range
- output rows are serialized with timezone information

## Source Row Schemas

### Email

Key fields:

- `email_id`
- `thread_id`
- `message_index_in_thread`
- `timestamp`
- sender fields
- participant fields
- `subject`
- `body`
- linked business ids
- label fields

### Slack

Key fields:

- `slack_message_id`
- `channel_id`
- `channel_name`
- thread fields
- `timestamp`
- `body`
- `mentions`
- `reactions`
- linked business ids
- label fields

### Teams

Key fields:

- `teams_message_id`
- `team_id`
- `channel_id`
- `chat_or_channel`
- `meeting_id`
- `body`
- `file_refs`
- linked business ids
- label fields

### Salesforce

Key fields:

- `salesforce_record_id`
- `object_type`
- `record_id`
- ownership and foreign-key style ids
- `subject`
- `text_body`
- `structured_fields`
- label fields

## Why The Domain Layer Exists

The project deliberately separates enterprise state from rendered records.

Without that separation:

- email text could say something unsupported by CRM state
- Slack blockers could appear without a real ticket
- attendance could be claimed without an event
- cross-system consistency would be fragile

With the domain layer, every rendered record has a place to point back to.
