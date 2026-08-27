"""
KikiFast Web UI — a simple dashboard for live control + deep session observability.

Runs as a daemon thread inside main.py (start_webui()), bound to the LAN on
:8090, completely OFF the speaking path. It only READS the in-memory
observability recorder and edits config.json via config_loader — it never blocks
a turn.

Tabs:
  - Controls : obvious switches for Unified Idle Mind, workers, summarization,
               vision, and routing. Each switch writes
               config.json, reloads it live, then RE-READS the value back so the
               UI always reflects the state that actually took effect.
  - Sessions : every speaking turn, Unified Idle Mind session, worker run, and
               summarization as its own card you
               can open to read end to end: the prompt/context fed, every LLM
               turn (prompt + response), every tool call (args + result), the
               final reply/appends. No more one flat firehose.
  - Live Feed: the raw chronological event stream (the firehose), filterable.
  - Context  : the exact context last fed to the local LLM.
  - Advanced : the full auto-generated config.json editor (power users).
"""

import threading

try:
    from flask import Flask, jsonify, request, Response
    _FLASK_OK = True
except Exception:                       # pragma: no cover - flask optional
    _FLASK_OK = False

from core.observability import get_recorder
from tools_and_config import config_loader

_PORT = 8090
_started = False
_status_provider = None     # optional callable -> dict of live flags


def set_status_provider(fn):
    global _status_provider
    _status_provider = fn


# --------------------------------------------------------------------------- #
#  Curated high-level controls (the "Controls" tab)                           #
#  Each control is a dotted config path + how to render it:                   #
#    kind="bool"          → On / Off          (value True/False)              #
#    kind="localcloud"    → Local / Cloud     (value True=Local / False=Cloud)#
#    kind="localcloud_str"→ Local / Cloud     (value "local" / "cloud")       #
#    kind="int"           → number box        (value int; optional "hint")    #
#    kind="csv"           → text box          (value list[str], shown a, b, c)#
#                                                                             #
#  Goal: expose nearly EVERYTHING you'd tune day-to-day here, grouped and     #
#  labelled in plain language, so the ambiguous raw "Advanced" editor is a    #
#  last resort. Cost knobs (cloud rate limits + per-feature caps) live up top.#
# --------------------------------------------------------------------------- #
_CONTROL_GROUPS = [
    {
        "title": "💸 Cloud cost guard — rate limits",
        "desc": "Hard ceiling on how many paid cloud calls (Gemini/Vertex/Groq) "
                "all background features may make. 0 = unlimited for that window. "
                "When a cap is hit the feature simply skips that call. The kill "
                "switch denies ALL background cloud calls in one toggle (Kiki can "
                "still talk via the local box).",
        "controls": [
            {"path": "cloud_limits.kill_switch", "label": "🛑 Disable ALL cloud calls (master kill switch)", "kind": "bool"},
            {"path": "cloud_limits.enabled", "label": "Enforce limits", "kind": "bool"},
            {"path": "cloud_limits.max_calls_per_hour", "label": "Max cloud calls / hour (all features)", "kind": "int"},
            {"path": "cloud_limits.max_calls_per_day", "label": "Max cloud calls / day (all features)", "kind": "int"},
            {"path": "cloud_limits.exempt_categories", "label": "Features exempt from ALL caps (comma-separated)", "kind": "csv"},
        ],
    },
    {
        "title": "💸 Per-feature cloud caps",
        "desc": "Finer cost control: cap each background feature on its own. "
                "0 = unlimited for that window — but the GLOBAL cap above still "
                "applies on top, so a feature is only truly uncapped if it is "
                "also listed in 'exempt from ALL caps'.",
        "controls": [
            {"path": "cloud_limits.per_category.idle_mind.per_hour", "label": "Unified idle mind / hour", "kind": "int"},
            {"path": "cloud_limits.per_category.idle_mind.per_day", "label": "Unified idle mind / day", "kind": "int"},
            {"path": "cloud_limits.per_category.vision.per_hour", "label": "Vision / hour", "kind": "int"},
            {"path": "cloud_limits.per_category.vision.per_day", "label": "Vision / day", "kind": "int"},
            {"path": "cloud_limits.per_category.summary.per_hour", "label": "Summaries / hour", "kind": "int"},
            {"path": "cloud_limits.per_category.summary.per_day", "label": "Summaries / day", "kind": "int"},
            {"path": "cloud_limits.per_category.face_enroll.per_hour", "label": "Face enrolment / hour", "kind": "int"},
            {"path": "cloud_limits.per_category.face_enroll.per_day", "label": "Face enrolment / day", "kind": "int"},
            {"path": "cloud_limits.per_category.reasoning.per_hour", "label": "Background reasoning / hour", "kind": "int"},
            {"path": "cloud_limits.per_category.reasoning.per_day", "label": "Background reasoning / day", "kind": "int"},
        ],
    },
    {
        "title": "👂 Always listen — ambient context",
        "desc": "Keep STT capture active. Buffered room speech is interpreted by the "
                "Unified Idle Mind, without a separate ambient cloud scheduler.",
        "controls": [
            {"path": "always_listen", "label": "Always listen", "kind": "bool"},
        ],
    },
    {
        "title": "🧠 Unified Idle Mind",
        "desc": "One cloud mind chooses whether to reflect, act, research lightly, "
                "research deeply, remember, journal, prepare one note, or do nothing.",
        "controls": [
            {"path": "idle_mind.enabled", "label": "Unified idle mind", "kind": "bool"},
            {"path": "idle_mind.max_sessions_per_day", "label": "Max sessions / day", "kind": "int"},
            {"path": "idle_mind.min_next_check_minutes", "label": "Earliest self-check (min)", "kind": "int"},
            {"path": "idle_mind.max_next_check_minutes", "label": "Latest self-check (min)", "kind": "int"},
            {"path": "idle_mind.light_investigative_calls", "label": "Light research calls", "kind": "int"},
            {"path": "idle_mind.deep_investigative_calls", "label": "Deep research calls", "kind": "int"},
            {"path": "idle_mind.max_persistence_calls", "label": "Max memory/journal writes per session", "kind": "int"},
            {"path": "idle_mind.max_action_calls", "label": "Max actions per session", "kind": "int"},
            {"path": "idle_mind.max_open_questions", "label": "Max open questions kept", "kind": "int"},
            {"path": "idle_mind.journal_max_entries", "label": "Max journal entries kept", "kind": "int"},
        ],
    },
    {
        "title": "👁️ Vision",
        "desc": "Periodic camera captures + scene/question generation.",
        "controls": [
            {"path": "vision_injection.enabled", "label": "Vision", "kind": "bool"},
            {"path": "use_local_llm.vision", "label": "Vision runs on", "kind": "localcloud"},
            {"path": "vision_injection.qa_interval_min_minutes", "label": "Vision Q&A min interval (min)", "kind": "int"},
            {"path": "vision_injection.qa_interval_max_minutes", "label": "Vision Q&A max interval (min)", "kind": "int"},
        ],
    },
    {
        "title": "⚙️ Workers",
        "desc": "Scheduled / event-triggered background agent tasks.",
        "controls": [
            {"path": "workers.enabled", "label": "Workers", "kind": "bool"},
            {"path": "workers.max_active_workers", "label": "Max active workers", "kind": "int"},
        ],
    },
    {
        "title": "📝 Context summarization",
        "desc": "Auto-compress the conversation when it gets long. A DENIED "
                "summarization is permanent memory loss — the conversation is "
                "compacted without one — which is why 'summary' ships in the "
                "cap-exempt list above.",
        "controls": [
            {"path": "agent.auto_summarize", "label": "Auto-summarize", "kind": "bool"},
            {"path": "use_local_llm.summary", "label": "Summaries run on", "kind": "localcloud"},
            {"path": "agent.token_limit", "label": "Summarize over N tokens", "kind": "int"},
            {"path": "agent.past_conversations_count", "label": "Past conversations remembered", "kind": "int"},
        ],
    },
    {
        "title": "🎭 Personality & roleplay",
        "desc": "A mode's prompt REPLACES Kiki's persona. Reinforcement repeats "
                "the character after the tool/memory blocks so it is not drowned "
                "out. Roleplay isolation additionally withholds Kiki's memory, "
                "skills, vision, face events, idle notes and peeping from that "
                "character — set per mode below.",
        "controls": [
            {"path": "assistant_modes.persona_reinforce.enabled", "label": "Repeat the character after tools/memory", "kind": "bool"},
            {"path": "assistant_modes.persona_reinforce.custom_prompt_modes_only", "label": "Only for modes with a custom prompt", "kind": "bool"},
            {"path": "assistant_modes.persona_reinforce.max_character_chars", "label": "Max character chars repeated", "kind": "int"},
        ],
    },
    {
        "title": "🔀 Cloud / local routing & delivery",
        "desc": "The Unified Idle Mind is always cloud-only so the local speaking "
                "slot remains untouched.",
        "controls": [
            {"path": "use_local_llm.reasoning", "label": "Reasoning runs on", "kind": "localcloud"},
        ],
    },
]


def _get_path(cfg, dotted):
    node = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def _roleplay_groups(cfg):
    """Per-mode roleplay toggles, built from config rather than hardcoded.

    Mode names are user-defined, so this group cannot live in the static
    _CONTROL_GROUPS list — a mode added by hand in config.json should show up
    here on the next page load.
    """
    modes = ((cfg.get("assistant_modes") or {}).get("modes") or {})
    controls = []
    for name in sorted(modes):
        if name.startswith("_") or not isinstance(modes[name], dict):
            continue  # `_*` keys are comments, not modes
        controls.append({
            "path": f"assistant_modes.modes.{name}.roleplay",
            "label": f"'{name}' is a roleplay character",
            "kind": "bool",
        })
    if not controls:
        return []
    return [{
        "title": "🎭 Roleplay character isolation",
        "desc": "ON = this mode is a different PERSON, so Kiki's world is kept "
                "out of its prompt: knowledge base, past conversations, skills, "
                "camera scene descriptions, face events, idle-mind notes and "
                "overheard speech are all suppressed, and its main_tools list "
                "should drop recall_memory/update_knowledge. Without this a "
                "character still gets ~10k chars briefing it on your life and "
                "talks about it. Leave OFF for Kiki herself and for senior mode, "
                "which genuinely need memory. Per-source overrides live under "
                "assistant_modes.modes.<name>.context in the Advanced editor.",
        "controls": controls,
    }]


def _controls_state():
    """Resolve every curated control's CURRENT live value from config."""
    cfg = config_loader.get_full_config()
    live = config_loader.LIVE_BLOCKS
    out = []
    for g in _CONTROL_GROUPS + _roleplay_groups(cfg):
        controls = []
        block_live = True
        for c in g["controls"]:
            top = c["path"].split(".", 1)[0]
            is_live = top in live
            block_live = block_live and is_live
            value = _get_path(cfg, c["path"])
            if c["kind"] == "bool" and value is None:
                # An absent flag (e.g. a mode with no `roleplay` key yet) is
                # OFF, not blank — otherwise the toggle renders unset.
                value = False
            controls.append({
                "path": c["path"], "label": c["label"], "kind": c["kind"],
                "value": value, "live": is_live,
            })
        out.append({"title": g["title"], "desc": g["desc"],
                    "live": block_live, "controls": controls})
    return out


# --------------------------------------------------------------------------- #
#  HTML (single page, no build step)                                          #
# --------------------------------------------------------------------------- #
_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kiki Dashboard</title>
<style>
:root{--bg:#0f1020;--panel:#181a2e;--panel2:#21243f;--line:#2a2d4a;--txt:#e8e9f3;
--mut:#8e92b8;--acc:#7c83ff;--ok:#3ddc91;--warn:#ffb454;--err:#ff6b6b;}
*{box-sizing:border-box}body{margin:0;font-family:'Segoe UI',system-ui,sans-serif;
background:var(--bg);color:var(--txt);font-size:14px}
header{display:flex;align-items:center;gap:14px;padding:12px 18px;background:var(--panel);
border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5;flex-wrap:wrap}
header h1{font-size:18px;margin:0;background:linear-gradient(135deg,#7c83ff,#b06ab3);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.pill{padding:3px 10px;border-radius:20px;background:var(--panel2);font-size:12px;color:var(--mut)}
.pill.on{color:var(--ok)}.tabs{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap}
.tab{padding:6px 14px;border-radius:8px;cursor:pointer;color:var(--mut);user-select:none}
.tab.active{background:var(--acc);color:#fff}
main{padding:16px;max-width:1100px;margin:0 auto}
.view{display:none}.view.active{display:block}
small.mut{color:var(--mut)}
/* controls */
.grp{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:14px}
.grp h2{font-size:15px;margin:0 0 2px}.grp .gd{color:var(--mut);font-size:12px;margin-bottom:10px}
.ctl{display:flex;align-items:center;gap:12px;padding:9px 0;border-top:1px solid var(--line)}
.ctl:first-of-type{border-top:none}.ctl .lab{flex:1;font-size:14px}
.seg{display:flex;border:1px solid #343862;border-radius:9px;overflow:hidden}
.seg button{background:var(--panel2);color:var(--mut);border:none;padding:7px 16px;cursor:pointer;font-size:13px;font-weight:600}
.seg button.on{background:var(--ok);color:#06281a}
.seg button.on.cloud{background:var(--acc);color:#fff}
.seg button.on.off{background:#3a3f63;color:var(--txt)}
.rb{font-size:10px;padding:2px 7px;border-radius:10px;background:rgba(255,180,84,.15);color:var(--warn)}
.numbox{width:110px;background:var(--panel2);color:var(--txt);border:1px solid #343862;border-radius:9px;padding:7px 10px;font-size:13px;text-align:right}
.csvbox{width:230px;background:var(--panel2);color:var(--txt);border:1px solid #343862;border-radius:9px;padding:7px 10px;font-size:13px}
.usageline{font-size:13px;color:var(--mut);padding:3px 0}.usageline b{color:var(--txt)}
.savemsg{margin-left:8px;color:var(--ok);font-size:12px;opacity:0;transition:opacity .2s}
.savemsg.show{opacity:1}
/* sessions */
.flt{display:flex;gap:8px;margin-bottom:12px;align-items:center;flex-wrap:wrap}
select,input,button{background:var(--panel2);color:var(--txt);border:1px solid #343862;
border-radius:8px;padding:7px 10px;font-size:13px}
button{cursor:pointer}button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
.sess{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--acc);
border-radius:10px;padding:10px 14px;margin-bottom:8px;cursor:pointer}
.sess:hover{background:var(--panel2)}
.sess .row{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.kind{font-weight:700;text-transform:capitalize}
.sname{color:var(--mut)}.stat{font-size:11px;padding:2px 8px;border-radius:10px;margin-left:auto}
.stat.running{background:rgba(124,131,255,.15);color:var(--acc)}
.stat.done{background:rgba(61,220,145,.15);color:var(--ok)}
.stat.failed{background:rgba(255,107,107,.15);color:var(--err)}
.smeta{font-size:12px;color:var(--mut);margin-top:5px}
.detail{margin-top:10px;display:none}.sess.open .detail{display:block}
.step{border-left:2px solid #343862;padding:4px 0 4px 12px;margin:8px 0}
.step .sh{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);font-weight:700}
.step.llm{border-left-color:var(--warn)}.step.tool{border-left-color:var(--ok)}
.step.context{border-left-color:var(--acc)}.step.reply{border-left-color:#b06ab3}
.step.result{border-left-color:#56ccf2}
pre{white-space:pre-wrap;word-break:break-word;background:#0c0d1c;border:1px solid var(--line);
border-radius:6px;padding:8px;margin:5px 0 0;font-size:12px;max-height:420px;overflow:auto}
.kv{font-size:12px;color:var(--mut)}.kv b{color:var(--txt)}
/* feed */
.ev{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc);
border-radius:8px;padding:8px 12px;margin-bottom:6px;cursor:pointer}
.ev:hover{background:var(--panel2)}.ev .row{display:flex;gap:10px;align-items:baseline}
.ev .ty{font-weight:600}.ev .nm{color:var(--mut)}.ev .du{margin-left:auto;color:var(--warn);font-size:12px}
.ev .ph{font-size:11px;color:var(--mut)}.ev pre{display:none}.ev.open pre{display:block}
/* cards / context / advanced */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.card h3{margin:0 0 8px;font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.card .big{font-size:26px;font-weight:600}.card .sub{font-size:12px;color:var(--mut);margin-top:4px}
.msg{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px}
.msg .role{font-size:11px;text-transform:uppercase;color:var(--acc);margin-bottom:4px}
.blk{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-bottom:12px;overflow:hidden}
.blk>summary{padding:12px 14px;cursor:pointer;font-weight:600;display:flex;align-items:center;gap:10px}
.badge{font-size:10px;padding:2px 7px;border-radius:10px}
.badge.live{background:rgba(61,220,145,.15);color:var(--ok)}
.badge.restart{background:rgba(255,180,84,.15);color:var(--warn)}
.field{display:grid;grid-template-columns:1fr 1.4fr;gap:10px;align-items:center;padding:6px 14px;border-top:1px solid #21243f}
.field label{color:var(--mut);font-size:12px;word-break:break-all}.field input,.field select{width:100%}
.bar{position:sticky;bottom:0;display:flex;gap:10px;align-items:center;padding:12px 16px;
background:var(--panel);border-top:1px solid var(--line);margin-top:14px}
.toast{margin-left:auto;color:var(--mut)}
</style></head><body>
<header>
  <h1>🤖 Kiki</h1>
  <span class="pill" id="status">…</span>
  <span class="pill" id="uptime"></span>
  <div class="tabs">
    <div class="tab active" data-v="ctrl">Controls</div>
    <div class="tab" data-v="sess">Sessions</div>
    <div class="tab" data-v="feed">Live Feed</div>
    <div class="tab" data-v="ctx">Context</div>
    <div class="tab" data-v="cfg">Advanced</div>
  </div>
</header>
<main>
  <section class="view active" id="ctrl">
    <small class="mut">Each switch saves &amp; applies immediately, then re-reads the value so you can confirm it took effect. <span class="rb">restart</span> = takes effect next restart.</small>
    <div id="ctrlbody" style="margin-top:12px"></div>
  </section>

  <section class="view" id="sess">
    <div class="flt">
      <select id="skind"><option value="all">all sessions</option></select>
      <label><input type="checkbox" id="sauto" checked> auto-refresh</label>
      <button id="srefresh">refresh</button>
      <small class="mut" id="scount"></small>
    </div>
    <div id="sesslist"></div>
  </section>

  <section class="view" id="feed">
    <div class="flt">
      <select id="ffilter"><option value="all">all events</option></select>
      <label><input type="checkbox" id="fauto" checked> auto-scroll</label>
      <small class="mut" id="fcount"></small>
    </div>
    <div id="feedlist"></div>
  </section>

  <section class="view" id="ctx">
    <small class="mut" id="ctxinfo"></small><div id="ctxlist"></div>
  </section>

  <section class="view" id="cfg">
    <small class="mut">Full config.json. High-level blocks apply live; <span class="badge restart">restart</span> blocks need a restart.</small>
    <div id="cfgform"></div>
    <div class="bar"><button class="primary" id="cfgsave">Save &amp; apply</button>
      <button id="cfgreset">Reset</button><span class="toast" id="cfgtoast"></span></div>
  </section>
</main>
<script>
const TYPE_COLORS={turn:'#7c83ff',tool:'#3ddc91',idle_mind:'#56ccf2',
worker:'#f2994a',vision:'#27ae60',
summarization:'#eb5757',ttfw:'#3ddc91'};
let lastId=0,dirty={},sessTimer=null,openSids=new Set();
const $=s=>document.querySelector(s),el=(t,c,h)=>{const e=document.createElement(t);
if(c)e.className=c;if(h!=null)e.innerHTML=h;return e;};
const esc=s=>(s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;');

document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');$('#'+t.dataset.v).classList.add('active');
  if(sessTimer){clearInterval(sessTimer);sessTimer=null;}
  if(t.dataset.v==='ctrl')loadControls();
  if(t.dataset.v==='sess'){loadSessions();sessTimer=setInterval(()=>{if($('#sauto').checked)loadSessions();},2000);}
  if(t.dataset.v==='ctx')loadCtx();
  if(t.dataset.v==='cfg')loadCfg();
});

/* ---------------- CONTROLS ---------------- */
async function loadControls(){
  const r=await(await fetch('/api/controls')).json();
  const body=$('#ctrlbody');body.innerHTML='';
  await renderUsage(body);
  r.groups.forEach(g=>{
    const card=el('div','grp');
    card.appendChild(el('h2',null,esc(g.title)));
    card.appendChild(el('div','gd',esc(g.desc)));
    g.controls.forEach(c=>{
      const row=el('div','ctl');
      row.appendChild(el('div','lab',esc(c.label)+(c.live?'':' <span class="rb">restart</span>')));
      if(c.kind==='int'){
        const inp=el('input');inp.type='number';inp.step='1';inp.min='0';
        inp.className='numbox';inp.value=(c.value==null?'':c.value);
        const commit=()=>{const v=parseInt(inp.value,10);
          if(!isNaN(v))setControl(c.path,v,row);};
        inp.onkeydown=(e)=>{if(e.key==='Enter')inp.blur();};
        inp.onblur=commit;
        row.appendChild(inp);
      }else if(c.kind==='csv'){
        const inp=el('input');inp.type='text';inp.className='csvbox';
        inp.placeholder='none';
        inp.value=Array.isArray(c.value)?c.value.join(', '):(c.value==null?'':c.value);
        // Sent as a LIST, not a string: cloud_budget reads exempt_categories as
        // a list, and round-tripping it as text would change the config type.
        const commit=()=>setControl(c.path,
          inp.value.split(',').map(s=>s.trim()).filter(Boolean),row);
        inp.onkeydown=(e)=>{if(e.key==='Enter')inp.blur();};
        inp.onblur=commit;
        row.appendChild(inp);
      }else{
        const seg=el('div','seg');
        let opts;
        if(c.kind==='bool'){
          opts=[['On',true,'on'],['Off',false,'off']];
        }else if(c.kind==='localcloud'){
          opts=[['Local',true,'on'],['Cloud',false,'cloud']];
        }else{ // localcloud_str
          opts=[['Local','local','on'],['Cloud','cloud','cloud']];
        }
        const cur=c.kind==='localcloud_str'?(c.value==='local'?'local':'cloud'):c.value;
        opts.forEach(([txt,val,cls])=>{
          const b=el('button',null,txt);
          if(cur===val)b.classList.add('on',cls);
          b.onclick=()=>setControl(c.path,val,row);
          seg.appendChild(b);
        });
        row.appendChild(seg);
      }
      row.appendChild(el('span','savemsg','✓ applied'));
      card.appendChild(row);
    });
    body.appendChild(card);
  });
}
async function renderUsage(body){
  let u;try{u=await(await fetch('/api/cloud_usage')).json();}catch(e){return;}
  if(!u)return;
  const card=el('div','grp');
  card.appendChild(el('h2',null,'📊 Cloud usage right now'));
  card.appendChild(el('div','gd',
    'Live count of paid cloud calls (resets on restart). '+
    (u.enabled?'Limits are ON.':'Limits are OFF — no ceiling.')));
  const g=u.global||{last_hour:0,last_day:0};
  card.appendChild(el('div','usageline',
    `<b>ALL features:</b> ${g.last_hour} in the last hour · ${g.last_day} today`));
  // Exempt calls happen but are not measured against the global cap, so show
  // the counted figure too — otherwise "171 calls" under a cap of 60 looks broken.
  if(g.counted_last_hour!=null && g.counted_last_hour!==g.last_hour){
    card.appendChild(el('div','usageline',
      `<b>counted toward the cap:</b> ${g.counted_last_hour}/hr · `+
      `${g.counted_last_day}/day <small class="mut">(exempt features excluded)</small>`));
  }
  const cats=u.categories||{};
  const names=Object.keys(cats).sort();
  if(names.length){
    let rows=names.map(n=>{const c=cats[n];
      const d=c.denied?` · <span style="color:var(--warn)">${c.denied} blocked</span>`:'';
      const x=c.exempt?' · <span style="color:var(--ok)">uncapped</span>':'';
      return `<div class="usageline">${esc(n)}: ${c.last_hour}/hr · ${c.last_day}/day${d}${x}</div>`;
    }).join('');
    card.appendChild(el('div',null,rows));
  }
  body.appendChild(card);
}
async function setControl(path,value,row){
  const res=await(await fetch('/api/config',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({updates:{[path]:value}})})).json();
  await loadControls();           // re-read => confirm the value that took effect
  if(res&&res.ok)toastFlash(res.restart_needed?'saved — restart needed for this':'applied live ✓');
  else toastFlash('error: '+(res&&res.error||'?'));
}
function toastFlash(txt){
  let t=$('#globaltoast');
  if(!t){t=el('div');t.id='globaltoast';t.style.cssText=
    'position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--panel2);'+
    'border:1px solid #343862;color:var(--txt);padding:8px 16px;border-radius:10px;z-index:20;font-size:13px';
    document.body.appendChild(t);}
  t.textContent=txt;t.style.opacity='1';
  clearTimeout(t._tm);t._tm=setTimeout(()=>t.style.opacity='0',1600);
}

/* ---------------- SESSIONS ---------------- */
function fmtTime(ts){return ts?new Date(ts*1000).toLocaleTimeString():'';}
function fmtDur(ms){if(ms==null)return'';return ms<1000?ms+'ms':(ms/1000).toFixed(1)+'s';}
async function loadSessions(){
  const k=$('#skind').value;
  const r=await(await fetch('/api/sessions?kind='+encodeURIComponent(k)+'&limit=100')).json();
  // keep kind filter populated
  const seen=new Set([...$('#skind').options].map(o=>o.value));
  (r.kinds||[]).forEach(kn=>{if(!seen.has(kn)){const o=el('option');o.value=o.textContent=kn;$('#skind').appendChild(o);}});
  const list=$('#sesslist');list.innerHTML='';
  (r.sessions||[]).forEach(s=>{
    const c=TYPE_COLORS[s.kind]||'#7c83ff';
    const card=el('div','sess');card.style.borderLeftColor=c;
    if(openSids.has(s.sid))card.classList.add('open');
    const metaBits=[];
    if(s.meta){if(s.meta.model)metaBits.push('model: '+s.meta.model);
      if(s.meta.tools_used&&s.meta.tools_used.length)metaBits.push('tools: '+s.meta.tools_used.join(', '));}
    card.innerHTML=`<div class="row">
      <span class="kind" style="color:${c}">${esc(s.kind.replace('_',' '))}</span>
      <span class="sname">${esc(s.name||'')}</span>
      <span class="stat ${s.status}">${s.status}${s.duration_ms!=null?' · '+fmtDur(s.duration_ms):''}</span>
      </div>
      <div class="smeta">${fmtTime(s.ts)} · ${s.step_count} steps${metaBits.length?' · '+esc(metaBits.join(' · ')):''}</div>
      <div class="detail" data-sid="${s.sid}"></div>`;
    card.onclick=(e)=>{
      if(e.target.closest('pre'))return;
      const open=card.classList.toggle('open');
      if(open){openSids.add(s.sid);loadSessionDetail(s.sid,card.querySelector('.detail'));}
      else openSids.delete(s.sid);
    };
    list.appendChild(card);
    if(openSids.has(s.sid))loadSessionDetail(s.sid,card.querySelector('.detail'));
  });
  $('#scount').textContent=(r.sessions||[]).length+' sessions';
}
async function loadSessionDetail(sid,box){
  const s=await(await fetch('/api/session/'+encodeURIComponent(sid))).json();
  if(!s||!s.sid){box.innerHTML='<small class="mut">(gone)</small>';return;}
  let h='';
  if(s.meta&&Object.keys(s.meta).length){
    h+='<div class="kv">';
    for(const k in s.meta){
      const v=s.meta[k];
      if(typeof v==='object')continue;
      h+=`<div><b>${esc(k)}:</b> ${esc(v)}</div>`;
    }
    h+='</div>';
    if(s.meta.prompt)h+=`<div class="step context"><div class="sh">prompt / context</div><pre>${esc(s.meta.prompt)}</pre></div>`;
  }
  (s.steps||[]).forEach(st=>{
    if(st.kind==='context'){
      h+=`<div class="step context"><div class="sh">context fed · ${esc(st.messages)} msgs</div><pre>${esc(st.content)}</pre></div>`;
    }else if(st.kind==='llm'){
      h+=`<div class="step llm"><div class="sh">llm turn ${esc(st.turn)}</div>`;
      if(st.prompt)h+=`<div class="kv">prompt</div><pre>${esc(st.prompt)}</pre>`;
      h+=`<div class="kv">response</div><pre>${esc(st.response)}</pre></div>`;
    }else if(st.kind==='tool'){
      h+=`<div class="step tool"><div class="sh">tool · ${esc(st.tool)}</div>`+
         `<div class="kv">args: ${esc(st.args)}</div><pre>${esc(st.result)}</pre></div>`;
    }else if(st.kind==='ttfw'){
      h+=`<div class="step"><div class="sh">time to first word: ${esc(st.ms)} ms</div></div>`;
    }else if(st.kind==='reply'){
      h+=`<div class="step reply"><div class="sh">spoken reply</div><pre>${esc(st.content)}</pre></div>`;
    }else if(st.kind==='result'){
      h+=`<div class="step result"><div class="sh">result</div><pre>${esc(st.content)}</pre></div>`;
    }else{
      h+=`<div class="step"><div class="sh">${esc(st.kind)}</div><pre>${esc(JSON.stringify(st,null,2))}</pre></div>`;
    }
  });
  box.innerHTML=h||'<small class="mut">no steps recorded</small>';
}
$('#srefresh').onclick=loadSessions;
$('#skind').onchange=loadSessions;

/* ---------------- LIVE FEED ---------------- */
function addEvents(evs){const list=$('#feedlist');const f=$('#ffilter').value;
  evs.forEach(e=>{lastId=Math.max(lastId,e.id);
    if(f!=='all'&&e.type!==f)return;
    const c=TYPE_COLORS[e.type]||'#7c83ff';
    const d=el('div','ev');d.style.borderLeftColor=c;
    const dur=e.duration_ms!=null?`${e.duration_ms} ms`:'';
    const ph=e.phase?`<span class="ph">${esc(e.phase)}</span>`:'';
    d.innerHTML=`<div class="row"><span class="ty" style="color:${c}">${esc(e.type)}</span>
      <span class="nm">${esc(e.name||'')}</span> ${ph}<span class="du">${dur}</span></div>
      <pre>${esc(JSON.stringify(e.meta||{},null,2)||'(no detail)')}</pre>`;
    d.onclick=()=>d.classList.toggle('open');list.appendChild(d);});
  while(list.children.length>500)list.removeChild(list.firstChild);
  if($('#fauto').checked)window.scrollTo(0,document.body.scrollHeight);
  $('#fcount').textContent=list.children.length+' shown';
}
async function poll(){try{
  const s=await(await fetch('/api/status')).json();
  $('#status').textContent=s.status||'running';$('#status').className='pill on';
  $('#uptime').textContent='up '+(s.uptime_h||'');
  const seen=new Set([...$('#ffilter').options].map(o=>o.value));
  (s.types||[]).forEach(t=>{if(!seen.has(t)){const o=el('option');o.value=o.textContent=t;$('#ffilter').appendChild(o);}});
  if($('#feed').classList.contains('active')){
    const ev=await(await fetch('/api/events?since='+lastId)).json();
    if(ev.events&&ev.events.length)addEvents(ev.events);
  }
}catch(e){$('#status').textContent='offline';$('#status').className='pill';}}
setInterval(poll,1500);poll();

/* ---------------- CONTEXT ---------------- */
async function loadCtx(){const c=await(await fetch('/api/context')).json();
  const info=$('#ctxinfo'),list=$('#ctxlist');list.innerHTML='';
  if(!c||!c.messages){info.textContent='No context captured yet.';return;}
  info.textContent='Captured '+fmtTime(c.ts)+' · '+c.messages.length+' messages';
  c.messages.forEach(m=>{const d=el('div','msg');
    d.innerHTML=`<div class="role">${esc(m.role)}</div><pre>${esc(m.content)}</pre>`;
    list.appendChild(d);});
}

/* ---------------- ADVANCED CONFIG ---------------- */
function renderNode(obj,path,parent){
  Object.keys(obj).forEach(k=>{const v=obj[k];const p=path?path+'.'+k:k;
    if(v&&typeof v==='object'&&!Array.isArray(v)){
      const det=el('details','');det.innerHTML=`<summary>${esc(k)}</summary>`;
      renderNode(v,p,det);parent.appendChild(det);
    }else{
      const row=el('div','field');const lab=el('label',null,esc(k));let inp;
      if(typeof v==='boolean'){inp=el('select');inp.innerHTML='<option value="true">true</option><option value="false">false</option>';inp.value=String(v);}
      else if(Array.isArray(v)){inp=el('input');inp.value=JSON.stringify(v);inp.dataset.json='1';}
      else if(typeof v==='number'){inp=el('input');inp.type='number';inp.step='any';inp.value=v;inp.dataset.num='1';}
      else{inp=el('input');inp.value=v;}
      inp.oninput=()=>{let val=inp.value;
        if(inp.dataset.num)val=parseFloat(val);
        else if(inp.tagName==='SELECT')val=(val==='true');
        else if(inp.dataset.json){try{val=JSON.parse(val);}catch(e){return;}}
        dirty[p]=val;};
      row.appendChild(lab);row.appendChild(inp);parent.appendChild(row);
    }});
}
async function loadCfg(){const r=await(await fetch('/api/config')).json();
  dirty={};const form=$('#cfgform');form.innerHTML='';
  Object.keys(r.config).forEach(blk=>{
    const live=r.live_blocks.includes(blk);
    const det=el('details','blk');
    det.innerHTML=`<summary>${esc(blk)}<span class="badge ${live?'live':'restart'}">${live?'live':'restart'}</span></summary>`;
    const body=el('div');renderNode(r.config[blk],blk,body);det.appendChild(body);form.appendChild(det);});
}
$('#cfgsave').onclick=async()=>{
  if(!Object.keys(dirty).length){$('#cfgtoast').textContent='no changes';return;}
  $('#cfgtoast').textContent='saving…';
  const res=await(await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({updates:dirty})})).json();
  $('#cfgtoast').textContent=res.ok?(res.restart_needed?'saved — RESTART needed for some fields':'saved & applied live ✓'):('error: '+res.error);
  if(res.ok)dirty={};
};
$('#cfgreset').onclick=()=>loadCfg();

loadControls();
</script></body></html>"""


# --------------------------------------------------------------------------- #
def _build_app():
    app = Flask(__name__)
    rec = get_recorder()

    @app.route("/")
    def index():
        return Response(_PAGE, mimetype="text/html")

    @app.route("/api/status")
    def status():
        st = rec.get_stats()
        out = {
            "status": "running",
            "uptime_s": st["uptime_s"],
            "uptime_h": f"{st['uptime_s'] // 3600}h {st['uptime_s'] % 3600 // 60}m",
            "types": sorted(st["counters"].keys()),
        }
        if _status_provider:
            try:
                out.update(_status_provider() or {})
            except Exception:
                pass
        return jsonify(out)

    @app.route("/api/events")
    def events():
        since = int(request.args.get("since", 0))
        etype = request.args.get("type", "all")
        limit = int(request.args.get("limit", 300))
        return jsonify({"events": rec.get_events(limit=limit, etype=etype, since_id=since)})

    @app.route("/api/stats")
    def stats():
        return jsonify(rec.get_stats())

    @app.route("/api/sessions")
    def sessions():
        kind = request.args.get("kind", "all")
        limit = int(request.args.get("limit", 80))
        return jsonify(rec.get_sessions(limit=limit, kind=kind))

    @app.route("/api/session/<sid>")
    def session_detail(sid):
        return jsonify(rec.get_session(sid) or {})

    @app.route("/api/context")
    def context():
        return jsonify(rec.get_latest_context() or {})

    @app.route("/api/controls")
    def controls():
        return jsonify({"groups": _controls_state()})

    @app.route("/api/cloud_usage")
    def cloud_usage():
        try:
            from core.cloud_budget import get_cloud_budget
            return jsonify(get_cloud_budget().snapshot())
        except Exception as e:
            return jsonify({"enabled": False, "error": str(e),
                            "global": {"last_hour": 0, "last_day": 0}, "categories": {}})

    @app.route("/api/config", methods=["GET", "POST"])
    def config_api():
        if request.method == "GET":
            return jsonify({
                "config": config_loader.get_full_config(),
                "live_blocks": sorted(config_loader.LIVE_BLOCKS),
            })
        data = request.get_json(force=True, silent=True) or {}
        updates = data.get("updates", {})
        if not isinstance(updates, dict) or not updates:
            return jsonify({"ok": False, "error": "no updates"})
        ok, err, restart = config_loader.apply_config_update(updates)
        get_recorder().record("config", name="update", phase="end",
                              keys=list(updates.keys()), restart_needed=restart)
        return jsonify({"ok": ok, "error": err, "restart_needed": restart})

    return app


def start_webui(port: int = _PORT, status_provider=None):
    """Start the dashboard in a daemon thread. Safe no-op if Flask is missing."""
    global _started
    if _started:
        return
    if not _FLASK_OK:
        print("[WebUI] Flask not installed — dashboard disabled")
        return
    if status_provider:
        set_status_provider(status_provider)
    _started = True

    def _run():
        try:
            app = _build_app()
            import logging
            logging.getLogger("werkzeug").setLevel(logging.ERROR)
            print(f"[WebUI] Dashboard at http://0.0.0.0:{port}")
            app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
        except Exception as e:
            print(f"[WebUI] failed to start: {e}")

    threading.Thread(target=_run, daemon=True, name="webui").start()
