import threading
import time
from typing import List, Optional

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common as mavlink
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "pymavlink is required for MAVLink control. Install it with `pip install pymavlink`."
    ) from exc

NEUTRAL_CHANNELS = (1500, 1500, 1500, 1500, 0, 0, 0, 0)
LAND_CHANNELS = (1500, 1500, 1300, 1500, 0, 0, 0, 0)
GUIDED_MODE_CUSTOM = 4
VELOCITY_TYPE_MASK = 0b110111000111

GESTURE_TO_GUIDED_VEL: dict[int, tuple[float, float, float]] = {
    0: (1.0, 0.0, 0.0),    # Forward => body-relative forward velocity
    1: (-1.0, 0.0, 0.0),   # Back => body-relative reverse velocity
    2: (0.0, 0.0, -0.6),   # Up => climb velocity
    3: (0.0, 0.0, 0.6),    # Down => descent velocity
    4: (0.0, 0.0, 0.0),    # Left => yaw rotation handled separately
    5: (0.0, 0.0, 0.0),    # Right => yaw rotation handled separately
    6: (0.0, 0.0, 0.0),    # Hold position -> zero velocity
}

GESTURE_TO_YAW_RATE: dict[int, float] = {
    4: -0.8,
    5: 0.8,
}

GESTURE_TO_RC: dict[int, tuple[int, int, int, int]] = {
    0: (1500, 1600, 1500, 1500),  # Forward => pitch forward
    1: (1500, 1400, 1500, 1500),  # Back => pitch backward
    2: (1500, 1500, 1600, 1500),  # Up => increase throttle
    3: (1500, 1500, 1400, 1500),  # Down => decrease throttle
    4: (1400, 1500, 1500, 1500),  # Left => roll left
    5: (1600, 1500, 1500, 1500),  # Right => roll right
    6: (1500, 1500, 1500, 1500),  # Stop => neutral controls
}

VOICE_MOVE_SPEED_MPS = 0.6
VOICE_MOVE_DURATION_S = 3.0
TAKEOFF_TIMEOUT_S = 5.0


class MavlinkGestureController:
    """Simple MAVLink controller for Pixhawk via a serial link."""

    def __init__(
        self,
        connection_string: str,
        baud: int = 57600,
        target_system: int = 1,
        target_component: int = 1,
        source_system: int = 255,
        heartbeat_timeout: float = 5.0,
    ):
        self.connection_string = connection_string
        self.baud = baud
        self.target_system = target_system
        self.target_component = target_component
        self.source_system = source_system
        self.master = mavutil.mavlink_connection(
            connection_string,
            baud=baud,
            source_system=source_system,
        )
        # pymavlink connections are not safe to read and write concurrently.
        # Serialize all socket access so telemetry is not corrupted while
        # climb and heartbeat messages are being sent from other threads.
        self._io_lock = threading.RLock()

        print(f"MAVLink: opening connection to {connection_string} at {baud} baud...")
        heartbeat = self._wait_for_heartbeat(heartbeat_timeout)
        if heartbeat is None:
            raise RuntimeError("No MAVLink heartbeat received.")

        self.target_system = getattr(heartbeat, "sysid", getattr(heartbeat, "srcSystem", self.target_system))
        self.target_component = getattr(heartbeat, "compid", getattr(heartbeat, "srcComponent", self.target_component))
        print(
            f"MAVLink: heartbeat received from system {self.target_system} component {self.target_component}.")
        self._last_command_id: Optional[int] = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="mavlink-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _wait_for_heartbeat(self, timeout: float):
        with self._io_lock:
            return self.master.wait_heartbeat(timeout=timeout)

    def _heartbeat_loop(self) -> None:
        """Keep the vehicle's GCS failsafe from expiring this control link."""
        while not self._heartbeat_stop.wait(1.0):
            try:
                with self._io_lock:
                    self.master.mav.heartbeat_send(
                        mavlink.MAV_TYPE_GCS,
                        mavlink.MAV_AUTOPILOT_INVALID,
                        0,
                        0,
                        mavlink.MAV_STATE_ACTIVE,
                    )
            except Exception as exc:
                print(f"MAVLink heartbeat failed: {exc}")
                return

    def close(self) -> None:
        heartbeat_stop = getattr(self, "_heartbeat_stop", None)
        heartbeat_thread = getattr(self, "_heartbeat_thread", None)
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat_thread is not None and heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=1.0)
        if self.master:
            try:
                self.send_velocity_command(0.0, 0.0, 0.0)
            except Exception as exc:
                print(f"[CLOSE ERROR] Failed to send Guided stop command: {exc}")
            try:
                self.send_neutral()
            except Exception:
                pass
            with self._io_lock:
                self.master.close()

    def _to_rc_channels(self, values: tuple[int, int, int, int]) -> List[int]:
        return [values[0], values[1], values[2], values[3], 0, 0, 0, 0]

    def send_rc_override(self, channel_values: tuple[int, int, int, int]) -> None:
        channels = self._to_rc_channels(channel_values)
        with self._io_lock:
            self.master.mav.rc_channels_override_send(
                self.target_system,
                self.target_component,
                channels[0],
                channels[1],
                channels[2],
                channels[3],
                channels[4],
                channels[5],
                channels[6],
                channels[7],
            )

    def send_neutral(self) -> None:
        with self._io_lock:
            self.master.mav.rc_channels_override_send(
                self.target_system,
                self.target_component,
                *NEUTRAL_CHANNELS,
            )

    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0) -> None:
        """Send a body-relative velocity setpoint in guided mode.

        `MAV_FRAME_BODY_OFFSET_NED` makes forward/back/left/right follow the drone's current heading,
        while vertical movement still follows up/down relative to the body.
        When no gesture is active, the caller should send (0,0,0) to hold position at the current altitude.
        """
        try:
            print(f"[GESTURE] Sending velocity: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}")
            with self._io_lock:
                self.master.mav.set_position_target_local_ned_send(
                    int(time.time() * 1000) & 0xFFFFFFFF,
                    self.target_system,
                    self.target_component,
                    mavlink.MAV_FRAME_BODY_OFFSET_NED,
                    VELOCITY_TYPE_MASK,
                    0.0,
                    0.0,
                    0.0,
                    vx,
                    vy,
                    vz,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    yaw_rate,
                )
        except Exception as e:
            print(f"[GESTURE ERROR] Failed to send velocity command: {e}")
            raise

    def move_distance(self, direction: str, distance_m: float = 0.5) -> None:
        """Move for the fixed voice-command interval, then explicitly stop."""
        if direction == "forward":
            velocity = (VOICE_MOVE_SPEED_MPS, 0.0, 0.0)
        elif direction == "back":
            velocity = (-VOICE_MOVE_SPEED_MPS, 0.0, 0.0)
        elif direction == "up":
            velocity = (0.0, 0.0, -VOICE_MOVE_SPEED_MPS)
        elif direction == "down":
            velocity = (0.0, 0.0, VOICE_MOVE_SPEED_MPS)
        elif direction == "left":
            velocity = (0.0, -VOICE_MOVE_SPEED_MPS, 0.0)
        elif direction == "right":
            velocity = (0.0, VOICE_MOVE_SPEED_MPS, 0.0)
        else:
            raise ValueError(f"Unsupported voice direction: {direction}")

        try:
            self.send_velocity_command(*velocity)
            time.sleep(VOICE_MOVE_DURATION_S)
        finally:
            self.send_velocity_command(0.0, 0.0, 0.0)


    def _wait_for_altitude(
        self,
        target_alt_m: float,
        timeout_s: float = TAKEOFF_TIMEOUT_S,
        tol: float = 0.2,
    ) -> bool:
        """Wait until vehicle reaches target altitude (relative) within tolerance.

        Checks GLOBAL_POSITION_INT.relative_alt (mm), which is the relative
        altitude used for the takeoff target.
        Returns True if reached, False on timeout.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            with self._io_lock:
                msg = self.master.recv_match(
                    type=['GLOBAL_POSITION_INT', 'LOCAL_POSITION_NED'],
                    blocking=False,
                )
            if msg is None:
                time.sleep(min(0.1, remaining))
                continue
            try:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    rel_mm = getattr(msg, 'relative_alt', None)
                    if rel_mm is not None:
                        rel_m = rel_mm / 1000.0
                        print(f"[TELEM] GLOBAL_POSITION_INT.relative_alt={rel_m:.3f} m")
                        if rel_m >= (target_alt_m - tol):
                            return True
            except Exception:
                pass
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def send_takeoff_command(
        self,
        altitude_m: float = 3.5,
        timeout_s: float = TAKEOFF_TIMEOUT_S,
    ) -> bool:
        """Attempt a guided takeoff using MAV_CMD_NAV_TAKEOFF.

        Returns True if telemetry indicates the vehicle reached the requested altitude, False otherwise.
        """
        try:
            print(f"[TAKEOFF] Sending MAV_CMD_NAV_TAKEOFF to {altitude_m} m")
            # MAV_CMD_NAV_TAKEOFF: param7 is altitude (relative)
            with self._io_lock:
                self.master.mav.command_long_send(
                    self.target_system,
                    self.target_component,
                    mavlink.MAV_CMD_NAV_TAKEOFF,
                    0,
                    0, 0, 0, 0, 0, 0,
                    float(altitude_m),
                )
        except Exception as e:
            print(f"[TAKEOFF ERROR] Failed to send MAV_CMD_NAV_TAKEOFF: {e}")
            return False

        # Wait for altitude to be reached
        success = self._wait_for_altitude(altitude_m, timeout_s=timeout_s)
        if success:
            print(f"[TAKEOFF] Reached target altitude ~{altitude_m} m")
        else:
            print(f"[TAKEOFF] Did NOT reach target altitude within timeout")
        return success

    def arm_and_enter_guided(self, custom_mode: int = GUIDED_MODE_CUSTOM, takeoff_altitude: float = 3.5) -> None:
        """Set guided mode, arm the vehicle, and perform a guided takeoff."""
        takeoff_deadline = time.monotonic() + TAKEOFF_TIMEOUT_S
        print("[ARM] Setting GUIDED mode...")
        with self._io_lock:
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                custom_mode,
                0,
                0,
                0,
                0,
                0,
            )
        time.sleep(0.2)
        print("[ARM] Arming drone...")
        with self._io_lock:
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        time.sleep(0.3)
        print("[ARM] Attempting guided takeoff to %.1f m using MAV_CMD_NAV_TAKEOFF..." % (takeoff_altitude,))
        try:
            # Reserve part of the five-second window for the fallback path.
            remaining_timeout = max(0.0, takeoff_deadline - time.monotonic())
            success = self.send_takeoff_command(
                takeoff_altitude,
                timeout_s=min(3.0, remaining_timeout),
            )
        except Exception as exc:
            print(f"[ARM] send_takeoff_command raised exception: {exc}")
            success = False

        if not success:
            fallback_timeout_s = max(0.0, takeoff_deadline - time.monotonic())
            fallback_deadline = takeoff_deadline
            print(
                "[ARM] MAV_CMD_NAV_TAKEOFF did not complete, falling back to "
                f"velocity-based takeoff until {takeoff_altitude:.1f} m "
                f"(remaining {fallback_timeout_s:.1f}s of {TAKEOFF_TIMEOUT_S:.0f}s total)"
            )
            iteration = 0
            while time.monotonic() < fallback_deadline:
                iteration += 1
                timestamp = int(time.time() * 1000) & 0xFFFFFFFF
                print(f"[TAKEOFF FALLBACK] sending climb velocity (iteration {iteration})")
                with self._io_lock:
                    self.master.mav.set_position_target_local_ned_send(
                        timestamp,
                        self.target_system,
                        self.target_component,
                        mavlink.MAV_FRAME_LOCAL_NED,
                        0x0DC7,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                remaining_timeout = max(0.0, fallback_deadline - time.monotonic())
                if self._wait_for_altitude(
                    takeoff_altitude,
                    timeout_s=min(0.1, remaining_timeout),
                ):
                    print("[ARM] Reached altitude during fallback")
                    success = True
                    break
                time.sleep(min(0.1, max(0.0, fallback_deadline - time.monotonic())))
            if not success:
                print("[ARM] Still did not reach altitude after fallback")

        print("[ARM] Sending hover command (0 velocity)")
        # After takeoff (successful or not), send hover/stop command
        self.send_velocity_command(0.0, 0.0, 0.0)
        if success:
            print("[ARM] Ready for gesture commands")
        else:
            print("[ARM] Not at target altitude — gestures may not cause lift until airborne")

    def send_land_command(self) -> None:
        with self._io_lock:
            self.master.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavlink.MAV_CMD_NAV_LAND,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

    def update_gesture(self, gesture_id: int) -> None:
        if gesture_id == 7:
            self.send_land_command()
            self._last_command_id = gesture_id
            return

        if gesture_id in GESTURE_TO_YAW_RATE:
            yaw_rate = GESTURE_TO_YAW_RATE[gesture_id]
            self.send_velocity_command(0.0, 0.0, 0.0, yaw_rate=yaw_rate)
            self._last_command_id = gesture_id
            return

        if gesture_id in GESTURE_TO_GUIDED_VEL:
            vx, vy, vz = GESTURE_TO_GUIDED_VEL[gesture_id]
            self.send_velocity_command(vx, vy, vz)
            self._last_command_id = gesture_id
            return

        if gesture_id in GESTURE_TO_RC:
            self.send_rc_override(GESTURE_TO_RC[gesture_id])
            self._last_command_id = gesture_id
            return

        self.send_velocity_command(0.0, 0.0, 0.0)
        self._last_command_id = None

    def get_last_command(self) -> Optional[int]:
        return self._last_command_id
