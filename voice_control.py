import json
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import speech_recognition as sr

try:
    from vosk import KaldiRecognizer, Model
except Exception:
    KaldiRecognizer = None
    Model = None


class VoiceControl:
    """Offline voice control for bounded movement and landing commands."""

    COMMAND_WORDS = ("go", "back", "up", "down", "right", "left", "land")
    VOSK_GRAMMAR = json.dumps(list(COMMAND_WORDS))
    MIN_COMMAND_SIMILARITY = 0.55
    MOVE_DISTANCE_M = 0.5

    def __init__(self, controller=None, on_recognize=None):
        self.controller = controller
        self.on_recognize = on_recognize
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self._stop = threading.Event()
        self._thread = None
        self.vosk_model = None
        self._vosk_available = False

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="voice-control",
            daemon=True,
        )
        self._thread.start()
        return True

    def set_controller(self, controller) -> None:
        self.controller = controller

    def _find_model_paths(self):
        bundled_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        executable_dir = Path(sys.executable).resolve().parent
        candidate_roots = [
            bundled_dir / "model",
            bundled_dir / "models" / "vosk",
            executable_dir / "model",
            executable_dir / "models" / "vosk",
            Path.cwd() / "model",
            Path.cwd() / "models" / "vosk",
        ]
        for base in candidate_roots:
            if not base.exists():
                continue
            if (base / "graph").exists() or (base / "conf").exists():
                yield base
                continue
            for child in sorted(base.iterdir()):
                if child.is_dir() and (
                    (child / "graph").exists() or (child / "conf").exists()
                ):
                    yield child

    def _load_model(self) -> None:
        if Model is None:
            print("VoiceControl: the vosk package is not installed.")
            return

        found_model = False
        for model_path in self._find_model_paths():
            found_model = True
            try:
                self.vosk_model = Model(str(model_path))
                self._vosk_available = True
                print(f"VoiceControl: Vosk model loaded from {model_path}")
                return
            except Exception as exc:
                print(f"VoiceControl: failed to load Vosk model from {model_path}: {exc}")

        if not found_model:
            print("VoiceControl: no Vosk model folder found.")
            print("VoiceControl: download vosk-model-small-en-us-0.15 and extract it")
            print("VoiceControl: into a model folder beside main.py.")

    def _listen_loop(self) -> None:
        try:
            self._load_model()
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.6
                self.recognizer.phrase_threshold = 0.2
                self.recognizer.non_speaking_duration = 0.3
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("VoiceControl: microphone calibrated. Awaiting commands...")

                while not self._stop.is_set():
                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=4,
                            phrase_time_limit=2,
                        )
                    except sr.WaitTimeoutError:
                        continue

                    command = self._recognize_command(audio)
                    if command:
                        print(f"VoiceControl: recognized command '{command}'")
                        if callable(self.on_recognize):
                            self.on_recognize(command)
                        self._execute_command(command)
        except Exception as exc:
            print(f"VoiceControl: microphone/listener stopped: {exc}")

    def _recognize_command(self, audio):
        if not self._vosk_available:
            return None
        try:
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            recognizer = KaldiRecognizer(
                self.vosk_model,
                16000,
                self.VOSK_GRAMMAR,
            )
            recognizer.AcceptWaveform(raw)
            result = json.loads(recognizer.FinalResult())
            return self._canonical_command(result.get("text", ""))
        except Exception as exc:
            print(f"VoiceControl: Vosk recognition failed: {exc}")
            return None

    @classmethod
    def _canonical_command(cls, text: str):
        words = " ".join(str(text).lower().split()).split()
        if not words:
            return None
        best_word = max(
            words,
            key=lambda word: max(
                SequenceMatcher(None, word, command).ratio()
                for command in cls.COMMAND_WORDS
            ),
        )
        best_command = max(
            cls.COMMAND_WORDS,
            key=lambda command: SequenceMatcher(None, best_word, command).ratio(),
        )
        similarity = SequenceMatcher(None, best_word, best_command).ratio()
        return best_command if similarity >= cls.MIN_COMMAND_SIMILARITY else None

    def _execute_command(self, command: str) -> None:
        controller = self.controller
        if controller is None:
            print(f"VoiceControl: recognized '{command}' (no MAV controller bound)")
            return
        try:
            if command == "land":
                controller.send_land_command()
                print("VoiceControl: land command sent")
                return
            direction = "forward" if command == "go" else command
            controller.move_distance(direction, distance_m=self.MOVE_DISTANCE_M)
            print(f"VoiceControl: moved {direction} {self.MOVE_DISTANCE_M:.1f} m")
        except Exception as exc:
            print(f"VoiceControl: command '{command}' failed: {exc}")

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
