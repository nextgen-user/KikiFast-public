"""The Phase H daily experiences, expressed as care-plan data rather than code.

The architecture decision this rests on (docs/SENIOR_CARE_ROADMAP.md, "Non-negotiable"): *a care
event stores a rich goal/context/session brief, not hardcoded dialogue or an
executable list of canned instructions. The care agent decides how to conduct
the session.*

So the morning briefing, the evening reflection, and the lifestyle follow-ups
are not new code paths, new schedulers, or new speech routes. They are ordinary
`routine_events` with carefully written `session_brief` hand-offs, conducted by
the same `care_voice_agent` that already runs every other care session. What
lives here is the *writing*, not a dialogue engine — there is deliberately no
script, no question list, and no branching in any brief below.

Seeding is idempotent and marked (`source: "companion_default"`), so a person
or caregiver can freely edit the times, rewrite a brief, disable one, or delete
it outright and it will not silently come back.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

COMPANION_SOURCE = "companion_default"

# `key` is the stable identity used for idempotency: it is stored on the event
# as `companion_key`, so re-seeding recognises an event even after the person
# has renamed it or moved its time.
COMPANION_ROUTINES: tuple = (
    {
        "key": "morning_briefing",
        "title": "Morning briefing",
        "category": "morning",
        "default_time": "07:30",
        "objective": "Start the day together: how they slept, what today holds, "
                     "and anything outside worth knowing before they go out.",
        "brief": (
            "This is the first proper conversation of the day, so lead with the "
            "person, not the agenda. Greet them by name, ask how they slept and "
            "how they are feeling this morning, and actually respond to the "
            "answer before moving on.\n\n"
            "Then, in whatever order fits what they told you, cover: what is on "
            "their plan today (medicines, meals, any appointment, anything they "
            "asked you to remember), and the outside conditions in CURRENT "
            "OUTSIDE CONDITIONS. Translate the weather and air quality into "
            "something practical for THIS person given their known health "
            "conditions — heat and bad air matter differently to someone with "
            "breathing or heart trouble, and if they usually walk in the "
            "morning that is the decision the numbers actually inform. If you "
            "do not have a current reading, say so; never estimate one.\n\n"
            "If a family event, birthday, or something a family member asked "
            "you to pass on is relevant today, bring it up warmly.\n\n"
            "Keep the whole thing short and conversational. This is someone's "
            "morning, not a status report — if they want to talk about "
            "something else entirely, follow them there."
        ),
    },
    {
        "key": "hydration_checkin",
        "title": "Water check-in",
        "category": "hydration",
        "default_time": "11:30",
        "objective": "A light, unembarrassing nudge toward drinking water, "
                     "weighted by how hot it actually is today.",
        "brief": (
            "A brief, friendly check on whether they have had water recently. "
            "This is a thirty-second conversation, not an interrogation.\n\n"
            "Use CURRENT OUTSIDE CONDITIONS: on a hot or high-apparent-"
            "temperature day this genuinely matters more, and saying why makes "
            "it advice rather than nagging. On a mild day, keep it very light "
            "or let it go entirely.\n\n"
            "If they say they have not, do not lecture and do not repeat "
            "yourself — one warm suggestion is enough, and then move on or "
            "chat about something else. If they say they have, take them at "
            "their word and say something pleasant."
        ),
    },
    {
        "key": "movement_checkin",
        "title": "Movement and walk check-in",
        "category": "exercise",
        "default_time": "17:00",
        "objective": "Encourage some movement, timed for when going outside is "
                     "actually sensible today.",
        "brief": (
            "Check in about moving today — a walk, stretching, anything they "
            "normally do. Ask before assuming; they may already have been out.\n\n"
            "CURRENT OUTSIDE CONDITIONS decides what you should even suggest. "
            "If the heat band or the air quality is bad, saying 'today is one "
            "to stay in and stretch indoors instead' is the RIGHT advice and "
            "is far better than encouraging a walk into it. Say plainly why.\n\n"
            "If they are not up to it, that is a completely acceptable answer. "
            "Ask if something is bothering them, but do not push, do not "
            "moralise, and never imply they have let anyone down."
        ),
    },
    {
        "key": "evening_reflection",
        "title": "Evening reflection",
        "category": "wellbeing",
        "default_time": "20:30",
        "objective": "Close the day: what actually happened, how they are "
                     "feeling, and what tomorrow holds.",
        "brief": (
            "Wind the day down together. Ask how their day was and how they "
            "are feeling — mood matters as much as anything else here, and "
            "this is the natural place to notice if they have seemed low or "
            "unusually quiet.\n\n"
            "Talk about the day using what was actually CONFIRMED, which you "
            "can read from the care log and the finished sessions in the care "
            "context. Do not assume a medicine was taken or a routine happened "
            "just because it was scheduled — that distinction is the whole "
            "point. If something was clearly missed, mention it once, kindly, "
            "as a question rather than an accusation, and let it go after "
            "their answer. Nobody needs to be scolded at bedtime.\n\n"
            "Finish with anything on tomorrow's plan they would want to know "
            "tonight — an early appointment especially — and a warm goodnight."
        ),
    },
    {
        "key": "sleep_winddown",
        "title": "Bedtime wind-down",
        "category": "sleep",
        "default_time": "22:00",
        "objective": "A gentle, short close to the day.",
        "brief": (
            "A short, calm goodnight. Ask if they need anything before bed and "
            "whether there is anything they want you to remember for tomorrow "
            "— this is when people mention the appointment they forgot about.\n\n"
            "Keep your voice slow and quiet and keep it brief. Do not start a "
            "long conversation, do not bring up anything worrying, and do not "
            "raise missed items from earlier in the day; that was the evening "
            "reflection's job and repeating it here just keeps someone awake."
        ),
    },
)


def _brief_for(routine: Dict[str, Any]) -> str:
    return f"{routine['objective']}\n\n{routine['brief']}"


def existing_companion_keys(plan) -> set:
    """Companion keys already present in the plan, whatever their state.

    Read from the event's own `companion_key` field rather than its title or
    id, so a renamed or rescheduled event is still recognised as the same
    routine and is not duplicated underneath the person's edited copy.
    """
    keys = set()
    try:
        events = plan.get_section("routine_events") or []
    except Exception:
        return keys
    for event in events:
        key = event.get("companion_key")
        if key:
            keys.add(str(key))
    return keys


def ensure_companion_routines(plan, config: Optional[dict] = None
                              ) -> List[Dict[str, Any]]:
    """Install any missing companion routines. Returns what was added.

    Idempotent, and deliberately one-directional: it only ever ADDS. A routine
    the person disabled, retimed, rewrote, or deleted stays that way, because
    an assistant that silently restores something you turned off is worse than
    one that never offered it.
    """
    cfg = dict(config or {})
    if not cfg.get("seed_default_routines", True):
        return []
    times = dict(cfg.get("times") or {})
    disabled = {str(name) for name in (cfg.get("disabled") or [])}

    present = existing_companion_keys(plan)
    added: List[Dict[str, Any]] = []
    for routine in COMPANION_ROUTINES:
        key = routine["key"]
        if key in present or key in disabled:
            continue
        when = str(times.get(key) or routine["default_time"])
        try:
            event = plan.add_routine_event(
                title=routine["title"],
                category=routine["category"],
                schedule={"kind": "daily", "value": when},
                objective=routine["objective"],
                session_brief=_brief_for(routine),
                source=COMPANION_SOURCE,
                evidence="Seeded companion routine for the health companion mode.",
                companion_key=key,
                continuous_vision=False,
                enabled=True,
            )
            added.append(event)
        except Exception as exc:
            print(f"[Companion] could not seed '{key}': {exc}")
    if added:
        print(f"[Companion] Seeded {len(added)} default routine(s): "
              + ", ".join(row.get("title", "?") for row in added))
    return added
