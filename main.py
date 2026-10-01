import argparse
import builtins
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("aerosign")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_DIR / "aerosign.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

_original_print = builtins.print

def _log_print(*args, **kwargs):
    msg = " ".join(str(arg) for arg in args)
    logger.info(msg)
    return _original_print(*args, **kwargs)

builtins.print = _log_print

try:
    import cv2
except ModuleNotFoundError as exc:
    print("Error: OpenCV is not installed in the current Python environment.")
    print("Activate your virtual environment and install dependencies before running this script.")
    print("Example:")
    print("  e:\\python\\gesture_env\\Scripts\\Activate.ps1")
    print("  python main.py")
    raise exc

from gesture_control import GestureBuffer, GESTURE_LABELS, HandGestureDetector
try:
    import voice_control
except ModuleNotFoundError:
    voice_control = None
    print("Note: voice_control not available in this environment — voice features disabled.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hand gesture detector for drone command mapping."
    )
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument(
        "--buffer-len",
        type=int,
        default=10,
        help="Number of frames used to stabilize gesture output",
    )
    parser.add_argument(
        "--roi-x",
        type=int,
        default=10,
        help="Top-left X coordinate of the tracked region of interest",
    )
    parser.add_argument(
        "--roi-y",
        type=int,
        default=80,
        help="Top-left Y coordinate of the tracked region of interest",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=400,
        help="Width and height of the square ROI used for hand tracking",
    )
    parser.add_argument(
        "--mavlink",
        type=str,
        default="udp:0.0.0.0:14550",
        help="MAVLink connection string (serial port, udp, tcp, etc.). Default is udp:0.0.0.0:14550. Leave empty to disable MAVLink.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=57600,
        help="Baud rate for MAVLink serial connections.",
    )
    return parser.parse_args()


def draw_status(image, fps: float) -> None:
    """Draw camera performance status on screen."""
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(image, "Press 'q' to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)


def draw_gesture_feed(image, gesture_label: str) -> None:
    """Render a single stable gesture label in the center of the frame."""
    if gesture_label == "None":
        return
    label_text = f"Gesture: {gesture_label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    text_size, _ = cv2.getTextSize(label_text, font, scale, thickness)
    x = (image.shape[1] - text_size[0]) // 2
    y = text_size[1] + 20
    cv2.rectangle(
        image,
        (x - 10, y - text_size[1] - 10),
        (x + text_size[0] + 10, y + 10),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(image, label_text, (x, y), font, scale, (255, 255, 255), thickness)


def draw_mavlink_status(image, status_text: str) -> Tuple[int, int, int, int]:
    """Draw the current MAVLink connection status on the video frame and return the text box rect."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text_size, _ = cv2.getTextSize(status_text, font, scale, thickness)
    x = 10
    y = image.shape[0] - 20
    rect = (x - 8, y - text_size[1] - 8, x + text_size[0] + 8, y + 8)
    cv2.rectangle(
        image,
        (rect[0], rect[1]),
        (rect[2], rect[3]),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(image, status_text, (x, y), font, scale, (0, 255, 0), thickness)
    return rect


def draw_control_mode_status(image, mode_text: str) -> Tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text = f"Mode: {mode_text.title()}"
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = image.shape[1] - text_size[0] - 10
    y = 30
    rect = (x - 8, y - text_size[1] - 8, x + text_size[0] + 8, y + 8)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (x, y), font, scale, (200, 200, 0), thickness)
    return rect


def draw_voice_toggle_status(image, enabled: bool) -> Tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    text = "ShowVoice: ON" if enabled else "ShowVoice: OFF"
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = image.shape[1] - text_size[0] - 10
    y = 60
    rect = (x - 8, y - text_size[1] - 8, x + text_size[0] + 8, y + 8)
    color = (0, 180, 0) if enabled else (80, 80, 80)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (x, y), font, scale, color, thickness)
    return rect


def draw_land_button(image, active: bool, landing_time: float, timeout: float = 4.0) -> Tuple[int, int, int, int]:
    """Draw a Land button on the video feed. If active==True and landing_time is recent, show 'Landing...'"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    now = time.time()
    if active and (now - landing_time) < timeout:
        text = "Landing..."
        color = (0, 120, 255)
    else:
        text = "Land"
        color = (0, 0, 200)
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = 10
    y = 60
    rect = (x - 8, y - text_size[1] - 8, x + text_size[0] + 8, y + 8)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (x, y), font, scale, color, thickness)
    return rect


def draw_takeoff_button(image, active: bool) -> Tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text = "Taking off..." if active else "Takeoff"
    color = (0, 180, 255) if active else (0, 180, 0)
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = 105
    y = 60
    rect = (x - 8, y - text_size[1] - 8, x + text_size[0] + 8, y + 8)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (x, y), font, scale, color, thickness)
    return rect


def draw_voice_feed(image, text: str) -> None:
    if not text:
        return
    label_text = f'Voice: "{text}"'
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(label_text, font, scale, thickness)
    x = (image.shape[1] - text_size[0]) // 2
    y = image.shape[0] - 60
    cv2.rectangle(image, (x - 10, y - text_size[1] - 10), (x + text_size[0] + 10, y + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, label_text, (x, y), font, scale, (255, 255, 255), thickness)


def is_point_in_rect(point_x: int, point_y: int, rect: Tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = rect
    return left <= point_x <= right and top <= point_y <= bottom


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = HandGestureDetector()
    buffer = GestureBuffer(buffer_len=args.buffer_len)

    mav_controller: Optional[object] = None
    mav_controller_lock = threading.Lock()
    voice_controller = None
    stop_event = threading.Event()
    connection_thread = None

    mavlink_target = args.mavlink.strip()
    mavlink_enabled = False
    mav_status = "MAVLink: Disabled" if mavlink_target else "MAVLink not configured"

    mavlink_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    control_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    voice_toggle_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    land_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    takeoff_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    control_mode = "gesture"  # or "voice"
    last_voice_text = ""
    last_voice_time = 0.0
    VOICE_DISPLAY_TIMEOUT = 3.0
    show_voice_overlay = True

    # landing UI/state
    landing_requested = False
    landing_time = 0.0
    LANDING_DISPLAY_TIMEOUT = 4.0
    takeoff_lock = threading.Lock()
    takeoff_in_progress = False

    def toggle_mavlink() -> None:
        nonlocal mavlink_enabled, mav_status, mav_controller
        if not mavlink_target:
            mav_status = "No MAVLink target configured"
            return

        mavlink_enabled = not mavlink_enabled
        if not mavlink_enabled:
            mav_status = "MAVLink: Disabled"
            with mav_controller_lock:
                if mav_controller is not None:
                    try:
                        mav_controller.close()
                    except Exception:
                        pass
                    mav_controller = None
        else:
            mav_status = "Trying to connect to MAVLink..."

    def _send_gesture_command(controller, gesture_id: int) -> None:
        """Send a gesture command to the MAV controller in a consistent way.

        - `7` (Land) will call `send_land_command()` if available.
        - other >=0 values call `update_gesture(gesture_id)`.
        - `-1` is treated as hover fallback (6) to keep vehicle stable by default.
        """
        if controller is None:
            return
        try:
            if gesture_id == 7:
                try:
                    controller.send_land_command()
                except Exception:
                    # fallback to sending a land-like gesture if available
                    try:
                        controller.update_gesture(7)
                    except Exception:
                        raise

            elif gesture_id >= 0:
                controller.update_gesture(gesture_id)
            else:
                # treat neutral as HOVER by default
                try:
                    controller.update_gesture(6)
                except Exception:
                    try:
                        controller.send_neutral()
                    except Exception:
                        pass
        except Exception:
            # let caller handle disconnects
            raise

    def start_takeoff() -> None:
        nonlocal takeoff_in_progress, mav_status
        with mav_controller_lock:
            controller = mav_controller
        if controller is None:
            print("Takeoff requested, but no MAVLink controller connected.")
            with mav_controller_lock:
                mav_status = "MAVLink: connect before takeoff"
            return
        with takeoff_lock:
            if takeoff_in_progress:
                return
            takeoff_in_progress = True

        def takeoff_worker() -> None:
            nonlocal takeoff_in_progress, mav_status
            try:
                print("Takeoff requested from video feed.")
                controller.arm_and_enter_guided(takeoff_altitude=3.5)
                with mav_controller_lock:
                    mav_status = "MAVLink connected"
                print("Takeoff sequence completed.")
            except Exception as exc:
                print(f"MAVLink takeoff failed: {exc}")
                with mav_controller_lock:
                    mav_status = "MAVLink: takeoff failed"
            finally:
                with takeoff_lock:
                    takeoff_in_progress = False

        threading.Thread(target=takeoff_worker, name="mavlink-takeoff", daemon=True).start()

    def on_recognize(text: str) -> None:
        nonlocal last_voice_text, last_voice_time
        text = str(text).strip().lower()
        if not text:
            return
        last_voice_text = text
        last_voice_time = time.time()

    def on_mouse(event, x, y, flags, param) -> None:
        nonlocal mav_controller, voice_controller, control_mode, voice_toggle_rect, show_voice_overlay, land_rect, takeoff_rect, landing_requested, landing_time, last_voice_text, last_voice_time
        if event != cv2.EVENT_LBUTTONUP:
            return
        if is_point_in_rect(x, y, mavlink_rect):
            toggle_mavlink()
            return
        if is_point_in_rect(x, y, control_rect):
            # flip mode between gesture and voice (mutually exclusive)
            if control_mode == "gesture":
                # enable voice mode (voice can run without MAVLink controller)
                try:
                    if voice_controller is None:
                        voice_controller = voice_control.VoiceControl(mav_controller, on_recognize=on_recognize)
                        voice_controller.start()
                        print("VoiceControl: started (controller may be None until MAVLink connects).")
                except Exception as e:
                    print(f"VoiceControl start failed: {e}")
                    return
                control_mode = "voice"
                # set MAVLink controller to hover when switching modes
                with mav_controller_lock:
                    if mav_controller is not None:
                        try:
                            _send_gesture_command(mav_controller, -1)
                        except Exception:
                            try:
                                mav_controller.send_neutral()
                            except Exception:
                                pass
            else:
                # switch to gesture: stop voice controller
                try:
                    if voice_controller is not None:
                        voice_controller.close()
                except Exception:
                    pass
                voice_controller = None
                control_mode = "gesture"
                # set MAVLink controller to hover when switching back to gesture
                with mav_controller_lock:
                    if mav_controller is not None:
                        try:
                            _send_gesture_command(mav_controller, -1)
                        except Exception:
                            try:
                                mav_controller.send_neutral()
                            except Exception:
                                pass

            print(f"Control mode switched to: {control_mode}")
            return

        if is_point_in_rect(x, y, takeoff_rect):
            start_takeoff()
            return

        # Manual land button click
        if is_point_in_rect(x, y, land_rect):
            # Try to send a land command through MAV controller (thread-safe)
            with mav_controller_lock:
                controller = mav_controller
            if controller is None:
                print("Manual land requested, but no MAVLink controller connected.")
            else:
                try:
                    controller.send_land_command()
                    landing_requested = True
                    landing_time = time.time()
                    print("Manual land command sent.")
                except Exception:
                    try:
                        controller.update_gesture(7)
                        landing_requested = True
                        landing_time = time.time()
                        print("Manual land (gesture fallback) sent.")
                    except Exception as e:
                        print(f"Manual land failed: {e}")
            return

        # voice overlay toggle click
        if is_point_in_rect(x, y, voice_toggle_rect):
            show_voice_overlay = not show_voice_overlay
            print(f"Show voice overlay: {show_voice_overlay}")

    cv2.namedWindow("Aerosign Gesture Control", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Aerosign Gesture Control", on_mouse)

    if mavlink_target:

        def connection_loop() -> None:
            nonlocal mav_controller, mav_status, voice_controller
            while not stop_event.is_set():
                with mav_controller_lock:
                    enabled = mavlink_enabled
                    current_controller = mav_controller

                controller_to_close = None
                if not enabled:
                    if current_controller is not None:
                        with mav_controller_lock:
                            if mav_controller is current_controller:
                                mav_controller = None
                                controller_to_close = current_controller
                        try:
                            if voice_controller is not None:
                                voice_controller.set_controller(None)
                        except Exception:
                            pass
                    if controller_to_close is not None:
                        try:
                            controller_to_close.close()
                        except Exception:
                            pass
                    with mav_controller_lock:
                        mav_status = "MAVLink: Disabled"
                elif current_controller is None:
                    with mav_controller_lock:
                        mav_status = "Trying to connect to MAVLink..."
                    try:
                        from mavlink_control import MavlinkGestureController
                        controller = MavlinkGestureController(
                            connection_string=mavlink_target,
                            baud=args.baud,
                        )
                        print(f"MAVLink: connected to {mavlink_target} @ {args.baud}")

                        # Do not hold mav_controller_lock during takeoff. The
                        # camera loop reads this lock on every frame.
                        with mav_controller_lock:
                            still_enabled = mavlink_enabled and mav_controller is None
                            if still_enabled:
                                mav_controller = controller
                                mav_status = "MAVLink connected"

                        if still_enabled:
                            try:
                                _send_gesture_command(controller, 6)
                            except Exception:
                                pass
                            try:
                                if voice_controller is not None:
                                    voice_controller.set_controller(controller)
                            except Exception:
                                pass
                        else:
                            controller.close()
                    except ModuleNotFoundError as exc:
                        with mav_controller_lock:
                            mav_status = "pymavlink not installed"
                        print(f"MAVLink error: {exc}")
                        return
                    except Exception as exc:
                        with mav_controller_lock:
                            mav_status = "MAVLink connection failed"
                        print(f"MAVLink connection failed: {exc}")
                stop_event.wait(2.0)

        connection_thread = threading.Thread(target=connection_loop, daemon=True)
        connection_thread.start()

    prev_time = time.time()
    last_display_label = "None"
    last_mav_gesture = -1
    last_mav_command_time = 0.0
    mav_velocity_resend_interval = 0.2
    gesture_loss_timeout = 0.5
    last_raw_gesture_time = time.time()
    stable_timeout = 0.7
    last_stable_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        x1 = max(0, min(args.roi_x, w - args.roi_size))
        y1 = max(0, min(args.roi_y, h - args.roi_size))
        x2 = min(w, x1 + args.roi_size)
        y2 = min(h, y1 + args.roi_size)

        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        now = time.time()

        if control_mode == "gesture":
            processed_roi, gesture_id = detector.process(roi)
            buffer.add_gesture(gesture_id)
            command_id = buffer.get_gesture()
            if gesture_id != -1:
                last_raw_gesture_time = now
            elif now - last_raw_gesture_time >= gesture_loss_timeout:
                command_id = -1
        else:
            # Don't process gestures in voice mode
            processed_roi = roi
            gesture_id = -1
            command_id = -1
        gesture_label = GESTURE_LABELS.get(command_id, "None")

        if command_id != -1:
            last_display_label = gesture_label
            last_stable_time = now
        elif now - last_stable_time > stable_timeout:
            last_display_label = "None"

        # Send commands only from the active control mode
        with mav_controller_lock:
            controller = mav_controller
        if controller is not None:
            if control_mode == "gesture":
                should_send_gesture = (
                    command_id != -1
                    and (
                        command_id != last_mav_gesture
                        or (
                            0 <= command_id <= 6
                            and now - last_mav_command_time >= mav_velocity_resend_interval
                        )
                    )
                )
                if should_send_gesture:
                    try:
                        _send_gesture_command(controller, command_id)
                        last_mav_gesture = command_id
                        last_mav_command_time = now
                    except Exception as exc:
                        print(f"MAVLink send failed: {exc}")
                        with mav_controller_lock:
                            mav_controller = None
                            mav_status = "Trying to connect to MAVLink..."
                        last_mav_gesture = -1
                elif command_id == -1:
                    # when no gesture is detected, ensure vehicle hovers (gesture id 6)
                    if last_mav_gesture != 6:
                        try:
                            _send_gesture_command(controller, 6)
                            last_mav_gesture = 6
                            last_mav_command_time = now
                        except Exception as exc:
                            print(f"MAVLink send failed: {exc}")
                            with mav_controller_lock:
                                mav_controller = None
                                mav_status = "Trying to connect to MAVLink..."
                            last_mav_gesture = -1
            else:
                # In voice mode, gestures do not send commands.
                pass

        dt = now - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        prev_time = now

        # copy the processed ROI back into the frame for display
        frame[y1:y2, x1:x2] = processed_roi

        # draw gesture label only when gesture mode is active
        if control_mode == "gesture":
            draw_gesture_feed(frame, last_display_label)
        draw_status(frame, fps)
        with mav_controller_lock:
            current_status = mav_status
        mavlink_rect = draw_mavlink_status(frame, current_status)
        control_rect = draw_control_mode_status(frame, control_mode)
        # draw manual land button and voice toggle
        land_rect = draw_land_button(frame, landing_requested, landing_time, LANDING_DISPLAY_TIMEOUT)
        with takeoff_lock:
            takeoff_active = takeoff_in_progress
        takeoff_rect = draw_takeoff_button(frame, takeoff_active)
        voice_toggle_rect = draw_voice_toggle_status(frame, show_voice_overlay)

        # show voice/gesture overlay for the active mode (if enabled)
        if show_voice_overlay and (time.time() - last_voice_time < VOICE_DISPLAY_TIMEOUT):
            draw_voice_feed(frame, last_voice_text)

        # clear landing_requested after timeout
        if landing_requested and (time.time() - landing_time) > LANDING_DISPLAY_TIMEOUT:
            landing_requested = False

        cv2.imshow("Aerosign Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord('v'):
            show_voice_overlay = not show_voice_overlay
            print(f"Show voice overlay toggled: {show_voice_overlay}")

    detector.close()
    stop_event.set()
    if connection_thread is not None:
        connection_thread.join(timeout=1.0)
    with mav_controller_lock:
        controller = mav_controller
    if controller is not None:
        try:
            controller.close()
        except Exception:
            pass
    # close voice controller if running
    try:
        if voice_controller is not None:
            voice_controller.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
