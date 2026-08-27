"""
Kiki Control Client - Async ZeroMQ client for controlling face recognition pipeline.

This client communicates with hailo_follower_webcam_only.py via ZeroMQ:
- Commands (REQ socket port 5555): Send commands to control the pipeline
- Events (SUB socket port 5556): Receive face detection events

Usage:
    import asyncio
    from kiki_control_client import KikiController
    
    async def main():
        controller = KikiController()
        await controller.connect()
        
        # Set neck movement off
        await controller.set_neck_movement(False)
        
        # Train a new person
        await controller.train_person("John")
        
        # Listen for events
        async for event in controller.listen_events():
            print(f"Event: {event}")
    
    asyncio.run(main())
"""

import asyncio
import json
from typing import AsyncIterator, Dict, Any, Optional, Callable
import zmq
import zmq.asyncio

# Default configuration (same as server)
ZMQ_CMD_PORT = 5555
ZMQ_EVENT_PORT = 5556
DEFAULT_HOST = "192.0.2.20"  # Change to localhost if running locally


class KikiController:
    """
    Async controller for the Kiki face recognition pipeline.
    
    Uses ZeroMQ for fast, async communication:
    - REQ socket for sending commands
    - SUB socket for receiving face events
    """
    
    def __init__(self, host: str = DEFAULT_HOST, cmd_port: int = ZMQ_CMD_PORT, event_port: int = ZMQ_EVENT_PORT):
        """
        Initialize the controller.
        
        Args:
            host: IP address of the machine running hailo_follower_webcam_only.py
            cmd_port: Port for command socket (default 5555)
            event_port: Port for event socket (default 5556)
        """
        self.host = host
        self.cmd_port = cmd_port
        self.event_port = event_port
        
        self.context: Optional[zmq.asyncio.Context] = None
        self.cmd_socket: Optional[zmq.asyncio.Socket] = None
        self.event_socket: Optional[zmq.asyncio.Socket] = None
        
        self._connected = False
        self._event_callbacks: list[Callable[[Dict[str, Any]], None]] = []
    
    async def connect(self) -> bool:
        """
        Establish connection to the face recognition pipeline.
        
        Returns:
            True if connection successful
        """
        # Release any prior (failed) attempt first. The face handler retries
        # connect() every 10s while the controller is down; without this, each
        # retry overwrites self.context/sockets and leaks the old context's I/O
        # thread + FDs, eventually exhausting process threads (pthread EAGAIN →
        # "Resource temporarily unavailable" everywhere, hung executors).
        self._cleanup_sockets()
        try:
            self.context = zmq.asyncio.Context()

            # Command socket (REQ pattern). LINGER=0 so close() never blocks on a
            # dead peer.
            self.cmd_socket = self.context.socket(zmq.REQ)
            self.cmd_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            self.cmd_socket.setsockopt(zmq.SNDTIMEO, 5000)
            self.cmd_socket.setsockopt(zmq.LINGER, 0)
            self.cmd_socket.connect(f"tcp://{self.host}:{self.cmd_port}")

            # Event socket (SUB pattern)
            self.event_socket = self.context.socket(zmq.SUB)
            self.event_socket.setsockopt(zmq.LINGER, 0)
            self.event_socket.connect(f"tcp://{self.host}:{self.event_port}")
            self.event_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all events

            # Test connection
            state = await self.get_state()
            if state:
                self._connected = True
                print(f"[KikiController] Connected to {self.host}")
                print(f"[KikiController] Current state: {state}")
                return True

        except Exception as e:
            print(f"[KikiController] Connection failed: {e}")

        # Connect failed — release the context/sockets we just created so the
        # retry loop doesn't leak them.
        self._cleanup_sockets()
        self._connected = False
        return False

    def _cleanup_sockets(self):
        """Close sockets + terminate context; idempotent, never raises.

        Sockets are closed BEFORE term() (with LINGER=0 set) so term() doesn't
        block, and all refs are nulled so a later connect() starts clean."""
        for sock in (self.cmd_socket, self.event_socket):
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
        try:
            if self.context is not None:
                self.context.term()
        except Exception:
            pass
        self.cmd_socket = None
        self.event_socket = None
        self.context = None

    async def disconnect(self):
        """Close all connections."""
        self._cleanup_sockets()
        self._connected = False
        print("[KikiController] Disconnected")
    
    async def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a command and wait for response.
        
        Args:
            command: Command dictionary
            
        Returns:
            Response dictionary
        """
        if not self.cmd_socket:
            raise RuntimeError("Not connected")
        
        await self.cmd_socket.send_json(command)
        response = await self.cmd_socket.recv_json()
        return response
    
    # ==================== Commands ====================
    
    async def get_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the current pipeline state.
        
        Returns:
            State dictionary with webcam, mode, neck_movement, training_in_progress
        """
        try:
            return await self._send_command({"cmd": "get", "state": True})
        except Exception as e:
            print(f"[KikiController] get_state error: {e}")
            return None
    
    async def set_webcam(self, enabled: bool) -> bool:
        """
        Enable or disable the webcam.
        
        Args:
            enabled: True to enable, False to disable
            
        Returns:
            True if command successful
        """
        response = await self._send_command({
            "cmd": "set",
            "webcam": "on" if enabled else "off"
        })
        return response.get("status") == "ok"
    
    async def set_mode(self, mode: str) -> bool:
        """
        Set the pipeline mode.
        
        Args:
            mode: "eval" for inference mode
            
        Returns:
            True if command successful
        """
        response = await self._send_command({
            "cmd": "set",
            "mode": mode
        })
        return response.get("status") == "ok"
    
    async def set_neck_movement(self, enabled: bool) -> bool:
        """
        Enable or disable neck (motor) movement tracking.
        
        Args:
            enabled: True to enable, False to disable
            
        Returns:
            True if command successful
        """
        response = await self._send_command({
            "cmd": "set",
            "neck_movement": "on" if enabled else "off"
        })
        return response.get("status") == "ok"
    
    async def set_target_person(self, person_name: str) -> bool:
        """
        Set the target person to track.
        
        The robot will follow this person when neck_movement is enabled.
        
        Args:
            person_name: Name of the person to track (must be trained in the system)
            
        Returns:
            True if command successful
        """
        response = await self._send_command({
            "cmd": "set",
            "target_person": person_name
        })
        return response.get("status") == "ok"
    
    async def look(self, direction: str, steps: Optional[int] = None) -> bool:
        """
        Explicit one-shot NECK turn for expression (left / right / center).

        Distinct from `set_neck_movement` (which toggles autonomous face-tracking):
        this deliberately rotates the neck stepper a fixed amount, e.g. to "turn
        away when hurt" or glance toward something. Implemented server-side by the
        Hailo controller's stepper driver.

        Args:
            direction: "left", "right", or "center"
            steps: optional stepper-step count (server default if None)

        Returns:
            True if command successful
        """
        cmd = {"cmd": "set", "look": direction}
        if steps is not None:
            cmd["look_steps"] = int(steps)
        response = await self._send_command(cmd)
        return response.get("status") == "ok"

    async def set_full_body_movement(self, enabled: bool) -> bool:
        """
        Enable or disable full body movement tracking mode.
        
        When enabled:
        - Stops face recognition pipeline
        - Starts YOLO body detection pipeline
        - Activates robot chassis motors to follow person's body
        
        When disabled:
        - Stops body detection and robot motors
        - Restarts face recognition pipeline
        
        Args:
            enabled: True to enable full body mode, False for face mode
            
        Returns:
            True if command successful
        """
        response = await self._send_command({
            "cmd": "set",
            "full_body": "on" if enabled else "off"
        })
        return response.get("status") == "ok"
    
    async def train_person(self, person_name: str) -> Dict[str, Any]:
        """
        Start training for a new person.
        
        This will:
        1. Capture 5 photos (1 second apart) from the raw camera feed
        2. Save to ~/Kiki/hailo-apps/.../train/<person_name>/
        3. Run the training process
        4. Emit a 'training_complete' event when done
        5. Switch back to eval mode
        
        Args:
            person_name: Name for the new person
            
        Returns:
            Response dictionary
        """
        response = await self._send_command({
            "cmd": "set",
            "mode": "train",
            "person": person_name
        })
        return response
    
    async def get_active_faces(self) -> list:
        """Faces the server currently sees (confirmed). Lets a just-connected
        client sync who is in front now, since face_detected is a one-shot PUB
        message that a late subscriber would have missed.

        Returns:
            List of {"track_id", "name", "confidence"} (may be empty).
        """
        try:
            resp = await self._send_command({"cmd": "get", "active_faces": True})
            if isinstance(resp, dict):
                return resp.get("active_faces", []) or []
        except Exception as e:
            print(f"[KikiController] get_active_faces error: {e}")
        return []

    async def get_face_crop(self) -> str:
        """Fetch a base64 JPEG crop of the face currently in view (largest), for
        the OLED face card. Returns '' if none is fresh."""
        try:
            resp = await self._send_command({"cmd": "get", "face_crop": True})
            if isinstance(resp, dict):
                return resp.get("image_b64", "") or ""
        except Exception as e:
            print(f"[KikiController] get_face_crop error: {e}")
        return ""

    async def rename_person(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a trained person (temp → real). LanceDB just updates the label
        (no retraining) and the server renames the train folder to match.

        Args:
            old_name: current (temporary) label
            new_name: the real name to store

        Returns:
            Response dictionary ({"status": "ok"|"error", ...})
        """
        return await self._send_command({
            "cmd": "set",
            "rename": {"old": old_name, "new": new_name},
        })

    async def merge_into(self, real_name: str, temp_name: str) -> Dict[str, Any]:
        """Merge a mistakenly-created temp identity into an existing REAL person.

        Used when a known person was mis-recognized as Unknown (too little
        training data): the server moves the temp's captured images into the real
        person's folder and re-trains them, then deletes the temp record.

        Args:
            real_name: the existing known person's label
            temp_name: the temp identity to fold in and remove

        Returns:
            Response dictionary
        """
        return await self._send_command({
            "cmd": "set",
            "merge": {"real": real_name, "temp": temp_name},
        })

    async def forget_person(self, person_name: str) -> Dict[str, Any]:
        """Delete a person's face record + train folder (undo a duplicate enroll).

        Args:
            person_name: label to forget

        Returns:
            Response dictionary
        """
        return await self._send_command({
            "cmd": "set",
            "forget_person": person_name,
        })

    # ==================== Events ====================
    
    def add_event_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add a callback function to be called when events are received.
        
        Args:
            callback: Function that takes an event dict as argument
        """
        self._event_callbacks.append(callback)
    
    async def listen_events(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator that yields face events.
        
        Yields:
            Event dictionaries, e.g.:
            - {"event": "face_detected", "track_id": 123, "name": "Alex", "confidence": 0.95}
            - {"event": "face_lost", "track_id": 123, "name": "Alex"}
            - {"event": "training_complete", "person": "John"}
            - {"event": "unknown_face_enrolled", "temp_name": "guest_...", "image_b64": "...", "timestamp": 172...}
            - {"event": "crowd_detected", "face_count": 3}
            - {"event": "person_renamed", "old": "guest_...", "new": "Raj"}
            - {"event": "person_merged", "real": "Alex", "temp": "guest_..."}
        """
        if not self.event_socket:
            raise RuntimeError("Not connected")
        
        while True:
            try:
                event = await self.event_socket.recv_json()
                
                # Call registered callbacks
                for callback in self._event_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        print(f"[KikiController] Callback error: {e}")
                
                yield event
                
            except zmq.ZMQError as e:
                print(f"[KikiController] Event receive error: {e}")
                await asyncio.sleep(0.1)
    
    async def listen_events_background(self) -> asyncio.Task:
        """
        Start listening for events in the background.
        
        Events will be passed to registered callbacks via add_event_callback().
        
        Returns:
            The background task (can be cancelled)
        """
        async def _listener():
            async for event in self.listen_events():
                pass  # Callbacks handle the events
        
        return asyncio.create_task(_listener())


# ==================== Convenience Functions ====================

async def quick_command(host: str, webcam: Optional[str] = None,
                        mode: Optional[str] = None,
                        neck_movement: Optional[str] = None,
                        person: Optional[str] = None,
                        look: Optional[str] = None,
                        look_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    Quick one-shot command without maintaining a connection.

    Args:
        host: Target host IP
        webcam: "on" or "off"
        mode: "eval" or "train"
        neck_movement: "on" or "off"  (toggle autonomous face-tracking)
        person: Person name (required if mode is train)
        look: explicit neck turn — "left" / "right" / "center"
        look_steps: optional stepper-step count for the explicit turn

    Returns:
        Response dictionary
    """
    # CRITICAL: this is a per-call context+socket and is invoked once PER neck
    # gesture (robot/neck.py spawns a thread → asyncio.run(quick_command)). Each
    # zmq.asyncio.Context() spawns an I/O thread + file descriptors. When the
    # controller is down, send/recv raises ("Resource temporarily unavailable" /
    # RCVTIMEO) — so cleanup MUST be in finally, or every failed gesture leaks a
    # thread+FDs until the process hits its thread limit and pthread_create fails
    # process-wide (EAGAIN), which then hangs ThreadPoolExecutor-based timeouts
    # (e.g. the web-search wait). LINGER=0 so close() never blocks on a dead peer.
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(f"tcp://{host}:{ZMQ_CMD_PORT}")

        cmd = {"cmd": "set"}
        if webcam:
            cmd["webcam"] = webcam
        if mode:
            cmd["mode"] = mode
            if mode == "train" and person:
                cmd["person"] = person
        if neck_movement:
            cmd["neck_movement"] = neck_movement
        if look:
            cmd["look"] = look
            if look_steps is not None:
                cmd["look_steps"] = int(look_steps)

        await socket.send_json(cmd)
        response = await socket.recv_json()
        return response
    finally:
        socket.close()
        context.term()


# ==================== Example Usage ====================

async def example_usage():
    """Example showing how to use the controller."""
    
    controller = KikiController(host="192.0.2.20")
    
    if not await controller.connect():
        print("Failed to connect!")
        return
    
    # Check current state
    state = await controller.get_state()
    print(f"Current state: {state}")
    
    # Disable neck movement
    await controller.set_neck_movement(False)
    print("Neck movement disabled")
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Enable neck movement
    await controller.set_neck_movement(True)
    print("Neck movement enabled")
    
    # Listen for face events (run in background)
    async def on_face_event(event):
        event_type = event.get("event")
        if event_type == "face_detected":
            print(f"🟢 New face: {event.get('name')} (ID: {event.get('track_id')}, confidence: {event.get('confidence'):.2f})")
        elif event_type == "face_lost":
            print(f"🔴 Face lost: {event.get('name')} (ID: {event.get('track_id')})")
        elif event_type == "training_complete":
            print(f"✅ Training complete for: {event.get('person')}")
    
    controller.add_event_callback(on_face_event)
    event_task = await controller.listen_events_background()
    
    # Keep running for a while
    print("Listening for face events for 30 seconds...")
    await asyncio.sleep(30)
    
    # Cleanup
    event_task.cancel()
    await controller.disconnect()


async def example_training():
    """Example showing how to train a new person."""
    
    controller = KikiController(host="192.0.2.20")
    
    if not await controller.connect():
        return
    
    # Start training
    print("Starting training for 'TestPerson'...")
    response = await controller.train_person("TestPerson")
    print(f"Training response: {response}")
    
    # Listen for training complete event
    async for event in controller.listen_events():
        if event.get("event") == "training_complete":
            print(f"Training complete for {event.get('person')}!")
            break
    
    await controller.disconnect()


if __name__ == "__main__":
    print("Kiki Control Client")
    print("=" * 40)
    print("Running example usage...")
    asyncio.run(example_usage())
