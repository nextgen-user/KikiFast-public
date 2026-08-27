import asyncio
import time
import random
from core.vision.camera import capture_photo_b64
from core.brain.generate_llm_resp import generate as generate_llm
from core import local_llm
from tools_and_config.config_loader import get_llm_config, get_use_local_llm_config
from pathlib import Path
_LLM_CFG = get_llm_config()
# Image recognition routing is controlled ONLY by use_local_llm.vision. It used
# to also be forced on by use_local_speaking (always true), so vision always ran
# on the single-slot box and competed with speaking. With vision=false it goes
# to cloud gemini-3-flash, leaving the box free + warm for speech.
# Read LIVE (not cached at import) so the Web UI Local/Cloud toggle takes effect
# on the next capture without a restart.
def _use_local_vision() -> bool:
    return get_use_local_llm_config().get("vision", False)

target_path = Path("~/Kiki/hailo-apps/hailo_apps/python/pipeline_apps/face_recognition/train")
target_path = target_path.expanduser()

class VisionHandler:
    def __init__(self, config, loop, stt_queue, sleep_checker):
        self.config = config
        self.loop = loop
        self.stt_queue = stt_queue
        self.sleep_checker = sleep_checker

        prompts_config = config.get("prompts", {})
        agent_config = config.get("agent", {})
        known_people = [item.name for item in target_path.iterdir() if item.is_dir()]
        
        self.vision_prefix = prompts_config.get("vision_update", {}).get("prefix", "[WHAT KIKI SEES]: ")
        self.vision_prompt = prompts_config.get("vision_update", {}).get("prompt", "Describe what you see.").format(known_people=known_people)
        
        self.question_min_interval = agent_config.get("question_min_interval_seconds", 120)
        self.question_max_interval = agent_config.get("question_max_interval_seconds", 180)
        
        self.last_question_time = time.time()
        self.next_question_interval = random.randint(
            min(self.question_min_interval, self.question_max_interval),
            max(self.question_min_interval, self.question_max_interval)
        )
        self.pending_vision_context = None
        # Set by main.py after the unified idle mind is constructed. It returns
        # a source-grounded speaking prompt, or None to suppress a low-value
        # autonomous interruption.
        self.proactive_prompt_fn = None
        # Set by main.py: appends context to message_history + rewarms the
        # prefix immediately (returns True), or returns False when a turn is
        # mid-flight → fall back to pending_vision_context.
        self.history_inject_fn = None

        # --- Separate timer for autonomous VISION QA (Kiki proactively comments
        # on/asks about what it sees). This is DISTINCT from capture+inject:
        #   - capture + silent inject  -> happens every convo (force_trigger)
        #   - vision QA (spoken)        -> only every qa_interval (default 30-60 min)
        vi_cfg = config.get("vision_injection", {})
        self.qa_min_interval = vi_cfg.get("qa_interval_min_minutes", 30) * 60
        self.qa_max_interval = vi_cfg.get("qa_interval_max_minutes", 60) * 60
        self.last_qa_time = time.time()
        self.next_qa_interval = random.randint(
            min(self.qa_min_interval, self.qa_max_interval),
            max(self.qa_min_interval, self.qa_max_interval)
        )

    async def run_vision_update(self, force_trigger=False, force_qa=False):
        if self.sleep_checker():
            print("[Vision] Skipped - Kiki is sleeping.")
            return

        vision_injection_cfg = self.config.get("vision_injection", {})
        traditional_enabled = vision_injection_cfg.get("traditional_context_enabled", True)

        now = time.time()
        # Vision QA (proactive spoken comment) is on its OWN slow timer (30-60 min).
        is_qa_time = (now - self.last_qa_time) >= self.next_qa_interval or force_qa

        # We proceed to capture+analyze if: forced (every-message capture), it's QA
        # time, or the legacy "traditional context" mode is on.
        if not traditional_enabled and not is_qa_time and not force_trigger:
            return

        try:
            print("[Vision] Triggering vision capture (post-LLM response)...")
            b64_image = await self.loop.run_in_executor(None, capture_photo_b64)
            if not b64_image:
                print("[Vision] Error: No image returned by camera.")
                return
            
            def _analyze():
                if _use_local_vision():
                    # Local mmproj vision on the single-slot box; aborts if the
                    # user speaks (preempt_background), so it never blocks a reply.
                    res = local_llm.generate_background(
                        prompt=self.vision_prompt, image_b64=b64_image,
                        max_tokens=256, temperature=0.5
                    )
                    if res:
                        return res
                    # Local box refused (conversation hot) or failed → cloud,
                    # so periodic questions still work during conversations.
                    print("[Vision] Local box unavailable/refused — falling back to cloud vision.")
                return generate_llm(self.vision_prompt, b64_image=b64_image, purpose="vision")

            analysis_result = await self.loop.run_in_executor(None, _analyze)
            if not analysis_result:
                # Most commonly: preempted by a new user request → focus on the
                # present, drop the stale image, no pending context injected.
                print("[Vision] No analysis (failed or preempted by user request).")
                return
                
            full_vision_context = f"{self.vision_prefix}{analysis_result}"
            print(f"[Vision] Context stored: {full_vision_context[:100]}...")
            
            # Record into shared vision history (for Workers)
            try:
                from core.workers.worker_brain import get_vision_history
                get_vision_history().record_vision(full_vision_context)
            except Exception:
                pass  # Workers module may not be initialized yet
            
            now = time.time()
            if is_qa_time:
                # Vision QA: Kiki proactively speaks about what it sees (rare).
                self.last_qa_time = now
                self.next_qa_interval = random.randint(
                    min(self.qa_min_interval, self.qa_max_interval),
                    max(self.qa_min_interval, self.qa_max_interval)
                )
                proactive_prompt = None
                if self.proactive_prompt_fn:
                    try:
                        proactive_prompt = self.proactive_prompt_fn(full_vision_context)
                    except Exception as e:
                        print(f"[Vision] Proactive source selection failed: {e}")
                if proactive_prompt:
                    print("[Vision] QA interval reached. Queuing source-grounded "
                          "autonomous comment.")
                    await self.stt_queue.put(("autonomous_vision", proactive_prompt))
                else:
                    print("[Vision] QA interval reached, but no worthwhile grounded "
                          "topic was found; staying quiet.")
            else:
                # Capture + SILENT inject: bake into history + warm prefix NOW
                # so the next turn's prefill is a cache hit; if a turn is
                # mid-flight, hold as pending for injection at the next turn.
                injected = False
                if self.history_inject_fn:
                    try:
                        injected = self.history_inject_fn(full_vision_context)
                    except Exception as e:
                        print(f"[Vision] History inject failed: {e}")
                if injected:
                    print(f"[Vision] Scene context appended to history (pre-warmed for next turn).")
                else:
                    self.pending_vision_context = full_vision_context
                    print(f"[Vision] Silently injected scene context for next turn.")
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Vision] Error: {e}")
