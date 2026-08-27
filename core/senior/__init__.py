"""Senior Citizen Mode addon.

An additive elderly-care companion layer for Kiki. Nothing in the existing
speaking/idle/worker machinery is rewritten — this package plugs a care plan
(``care_plan.json``) into the existing workers scheduler (reminders, guided
exercises, daily summaries) and adds voice-first care-plan editing plus family
email alerts. Active only while the ``senior`` assistant mode is selected.
"""
