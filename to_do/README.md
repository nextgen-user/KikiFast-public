# `to_do/` — Parked code (NOT dead, NOT deleted)

This folder holds working code that has been **moved out of the active codebase** but
deliberately **kept** so it can be reintegrated later. Nothing here is imported by the
running app. Treat it as a freezer, not a trash can.

Reason for parking: Kiki was redefined as a **stationary unit whose only motion is neck
rotation (left/right/center)**. The wheeled mecanum chassis / locomotion stack was removed
from the live path but preserved here in case wheels are added back.

## Contents

| File | What it is | How to reintegrate |
|---|---|---|
| `motor_control.py` | Low-level gpiod + SoftPWM mecanum wheel driver (pins/trims/PWM). Was `robot/motor_control.py`. | Move back to `robot/`. Note: the Hailo process at `~/Kiki/navigation/` has its OWN copy of `motor_control.py`; this one was the Pi-side copy used by `movement.py`/`pet_test.py`. |
| `movement.py` | Chassis movement-tag parser (`<turn(90)>`, `<forward(50)>`, …) + `execute_movements()` (direct-GPIO). Was `robot/movement.py`. Superseded on the speaking path by `robot/neck.py` (neck tags). | Re-add the import in `main.py` and re-wire the tag plumbing if wheels return. |
| `pet_test.py` | Standalone chassis "pet motion" test harness. Was `robot/pet_test.py`. | Run directly; needs `motor_control.py` on the path. |
| `chassis_tools.py` | The chassis tool layer pulled out of `tools_and_config/tools.py`: `KikiMotorClient` (ZMQ client to the motor server on :5557), `VALID_MOTOR_ACTIONS`, and the `move`, `dance`, `follow_me` tools + the music ultrasonic safety monitor. | See the header in that file — re-add each function's TOOLS schema and `_ASYNC_TOOL_HANDLERS` entry in `tools.py`, and restore the `_music_safety_monitor` thread in `play_music`. |

## What stayed live (neck / gaze)
- `kiki_control_client.py` — `KikiController` (neck `set_neck_movement`, `set_target_person`,
  new explicit `look()`), unchanged ZMQ face/neck channel (:5555/:5556).
- `tools_and_config/tools.py` — `track_person` (neck gaze) stayed.
- `robot/neck.py` — NEW: `<neck:left|right|center>` expressive-gesture tags on the speaking path.
- `set_neck_active()` (formerly the mis-named `set_motor_relay`) — neck-tracking power toggle.

## `orphan_scripts/` — parked standalone scripts (0 importers)
Old experiment / scratch / test scripts that nothing in the app imported:
`oled_cam.py`, `oledd.py`, `oled_stream.py`, `oled_color_test.py` (OLED experiments),
`ttea.py`, `tes2.py`, `test.py`, `testsss.py`, `adba.py`, `lcdt.py` (scratch/test),
`slouch_detector.py` (standalone posture demo), and `sleep_backup.py` (near-duplicate of
`sound_effects/audioeffects/sleep.py`, references the old `/srv/kikifast` paths).
Run them directly if needed; none are wired into `main.py`.

## `debris/` — non-code artifacts
Debug images and a test wav left at the repo root: `debug_posture.jpg`,
`debug_webcam_1bit.png`, `debug_webcam_original.jpg`, `test_demo.wav`.

## Deliberately KEPT in place (not parked)
- `core/brain/big_brain.py` `BigBrainSuggestionManager` — write-only since reflection was
  merged (nothing reads its rolling buffer; the live path uses `to_prompt_injection()`
  directly). Left intact as a re-wireable feature per the user's request.
- `sound_effects/audioeffects/` wav/mp3/otf assets + `sleep.py`/`awake.py` — left untouched
  (filler/effects area under reconsideration).
