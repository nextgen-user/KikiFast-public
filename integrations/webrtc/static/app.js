/* KikiFast phone client.
   Captures camera + mic (echo-cancelled), streams to the RPi over WebRTC,
   plays RPi audio on the speaker, and ships sensor data over a DataChannel. */

const $ = (id) => document.getElementById(id);

let pc = null;
let dc = null;
let localStream = null;
let wakeLock = null;
let facing = "environment";   // 'environment' = back, 'user' = front
let micOn = true;

const statusEl = $("status");
function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "pill" + (cls ? " " + cls : "");
}

/* ---------- media capture (echo cancellation lives HERE, on the phone) ---- */
async function getMedia() {
  const constraints = {
    audio: {
      echoCancellation: true,     // kills speaker->mic echo
      noiseSuppression: true,     // noise cancellation
      autoGainControl: true,
      channelCount: 1,
      sampleRate: 48000,
    },
    video: {
      facingMode: { ideal: facing },
      // Small frame = low decode cost on the Pi + low latency.
      width:  { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 20, max: 24 },
    },
  };
  return navigator.mediaDevices.getUserMedia(constraints);
}

/* ---------- WebRTC ---------- */
async function connect() {
  pc = new RTCPeerConnection({ iceServers: [] }); // same USB link, no STUN needed

  // remote audio (RPi speaker output) -> <audio>
  pc.ontrack = (e) => {
    if (e.track.kind === "audio") $("remote").srcObject = e.streams[0];
  };

  pc.onconnectionstatechange = () => {
    const s = pc.connectionState;
    if (s === "connected") setStatus("live ●", "ok");
    else if (s === "failed" || s === "disconnected") setStatus(s, "bad");
    else setStatus(s);
  };

  // DataChannel for sensors + button events (cheap, no media track)
  dc = pc.createDataChannel("ctrl", { ordered: false, maxRetransmits: 0 });
  dc.onopen = () => startSensors();

  localStream.getTracks().forEach((t) => pc.addTrack(t, localStream));

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const res = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
  });
  const answer = await res.json();
  await pc.setRemoteDescription(answer);
}

function send(obj) {
  if (dc && dc.readyState === "open") {
    try { dc.send(JSON.stringify(obj)); } catch (_) {}
  }
}

/* ---------- sensors: gyro + accel + magnetometer/compass ---------- */
let sensorTimer = null;
let motion = { gx: 0, gy: 0, gz: 0, ax: 0, ay: 0, az: 0 };
let heading = null;   // compass / magnetometer-derived

function startSensors() {
  // gyroscope (rotationRate) + accelerometer
  window.addEventListener("devicemotion", (e) => {
    const r = e.rotationRate || {};
    const a = e.accelerationIncludingGravity || {};
    motion = {
      gx: r.alpha || 0, gy: r.beta || 0, gz: r.gamma || 0,   // deg/s
      ax: a.x || 0, ay: a.y || 0, az: a.z || 0,              // m/s^2
    };
  });

  // magnetometer / compass heading
  window.addEventListener("deviceorientationabsolute", onOrient);
  window.addEventListener("deviceorientation", onOrient);
  function onOrient(e) {
    if (e.webkitCompassHeading != null) heading = e.webkitCompassHeading;
    else if (e.alpha != null) heading = 360 - e.alpha; // alpha = magnetometer-fused
  }

  // raw magnetometer if the device exposes the Generic Sensor API
  try {
    if ("Magnetometer" in window) {
      const mag = new Magnetometer({ frequency: 20 });
      mag.addEventListener("reading", () => {
        motion.mx = mag.x; motion.my = mag.y; motion.mz = mag.z; // microtesla
      });
      mag.start();
    }
  } catch (_) {}

  // push a consolidated packet ~20x/sec
  sensorTimer = setInterval(() => {
    send({ t: "sensor", ts: Date.now(), heading, ...motion });
    $("sensorReadout").textContent =
      `gyro ${motion.gx.toFixed(0)},${motion.gy.toFixed(0)},${motion.gz.toFixed(0)}` +
      (heading != null ? ` | ${heading.toFixed(0)}°` : "");
  }, 50);
}

/* ---------- iOS-style motion permission (harmless on Android) ---------- */
async function requestMotionPermission() {
  const D = window.DeviceMotionEvent;
  if (D && typeof D.requestPermission === "function") {
    try { await D.requestPermission(); } catch (_) {}
  }
  const O = window.DeviceOrientationEvent;
  if (O && typeof O.requestPermission === "function") {
    try { await O.requestPermission(); } catch (_) {}
  }
}

/* ---------- screen wake lock (keep phone awake while streaming) ---------- */
async function keepAwake() {
  try {
    wakeLock = await navigator.wakeLock.request("screen");
    wakeLock.addEventListener("release", () => {});
  } catch (_) {}
}
document.addEventListener("visibilitychange", async () => {
  if (wakeLock === null && document.visibilityState === "visible") keepAwake();
});

/* ---------- fullscreen ---------- */
async function goFullscreen() {
  const el = document.documentElement;
  try { await (el.requestFullscreen || el.webkitRequestFullscreen).call(el); }
  catch (_) {}
}

/* ---------- camera flip (renegotiates the video sender) ---------- */
async function flipCamera() {
  facing = facing === "environment" ? "user" : "environment";
  const newStream = await getMedia();
  const newVideo = newStream.getVideoTracks()[0];
  const sender = pc.getSenders().find((s) => s.track && s.track.kind === "video");
  if (sender) await sender.replaceTrack(newVideo);
  localStream.getVideoTracks().forEach((t) => t.stop());
  // swap video track in the local stream/preview, keep the mic track
  localStream.removeTrack(localStream.getVideoTracks()[0]);
  localStream.addTrack(newVideo);
  $("preview").srcObject = localStream;
}

/* ---------- buttons ---------- */
function wireButtons() {
  document.querySelectorAll(".mock").forEach((btn) => {
    btn.addEventListener("click", () => send({ t: "event", name: btn.dataset.name }));
  });

  $("flipBtn").addEventListener("click", flipCamera);
  $("fsBtn").addEventListener("click", goFullscreen);

  $("muteBtn").addEventListener("click", () => {
    micOn = !micOn;
    localStream.getAudioTracks().forEach((t) => (t.enabled = micOn));
    $("muteBtn").textContent = micOn ? "🎙️ Mute" : "🔇 Unmute";
    $("muteBtn").classList.toggle("active", !micOn);
  });
}

/* ---------- start ---------- */
async function start() {
  $("startBtn").disabled = true;
  try {
    await requestMotionPermission();
    localStream = await getMedia();
    $("preview").srcObject = localStream;

    $("gate").classList.add("hidden");
    $("stage").classList.remove("hidden");

    await keepAwake();
    await goFullscreen();
    wireButtons();
    await connect();
  } catch (err) {
    setStatus("error: " + err.message, "bad");
    alert("Could not start: " + err.message +
          "\n\nMake sure you opened this over http://localhost (adb reverse) " +
          "so the camera/mic are allowed.");
    $("startBtn").disabled = false;
  }
}

$("startBtn").addEventListener("click", start);
