"""Cross-script keyboard state and listener helpers for UF LeRobot scripts.

This module centralises the duplicated keyboard handling found in
`uf_robot_teleop.py`, `uf_lerobot_record.py` and `uf_lerobot_eval.py`.
"""

from __future__ import annotations

import logging
import sys
import contextlib
from dataclasses import dataclass, field
from typing import Callable, Optional

from lerobot.utils.keyboard_input import TerminalKeyListener, pynput_listener_is_trusted, pynput_can_capture, keyboard


@dataclass
class KeyboardState:
    """Holds shared keyboard state across the teleop/eval/record scripts.

    Attributes:
        key_dict: Mapping from normalised key name (e.g. ``"key.esc"``) to
            whether the key is currently held down.
        events: Script-specific event flags (e.g. ``"exit"``, ``"reset"``,
            ``"stop_recording"``).  Callers decide which flags exist.
    """

    key_dict: dict[str, bool] = field(default_factory=dict)
    events: dict[str, bool] = field(default_factory=dict)

    def is_pressed(self, key: str) -> bool:
        """Return whether ``key`` is currently held down."""
        return self.key_dict.get(key, False)

    def set_event(self, name: str, value: bool = True) -> None:
        """Set an event flag."""
        self.events[name] = value
        self.reset_keys()

    def get_event(self, name: str) -> bool:
        """Return the value of an event flag (default ``False``)."""
        return self.events.get(name, False)

    def reset_keys(self) -> None:
        """Release all tracked keys."""
        for k in self.key_dict:
            if k != 'key.esc':
                self.key_dict[k] = False

OnPressCallback = Callable[[str, KeyboardState], None]
OnReleaseCallback = Callable[[str, KeyboardState], None]


def _register_keyboard_listener(on_press_key: Optional[Callable[[str], None]] = None,
                               on_release_key: Optional[Callable[[str], None]] = None):
    """Register a cross-platform keyboard listener.

    Returns the underlying listener object (pynput Listener or TerminalKeyListener),
    or ``None`` if no keyboard backend is available.
    """
    if pynput_can_capture() and keyboard is not None:
        if on_press_key:
            def on_press(key):
                with contextlib.suppress(Exception):
                    on_press_key(str(key).lower())
        else:
            on_press = None
        if on_release_key:
            def on_release(key):
                with contextlib.suppress(Exception):
                    on_release_key(str(key).lower())
        else:
            on_release = None
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        if pynput_listener_is_trusted(listener):
            logging.info("Keyboard listener started.")
            return listener
        # macOS without Accessibility / Input-Monitoring permission: the listener never
        # fires. Stop it and fall through to the terminal backend.
        logging.warning(
            "pynput keyboard listener is not trusted (missing macOS Accessibility / "
            "Input Monitoring permission); falling back to terminal keyboard input."
        )
        listener.stop()

    if sys.stdin.isatty() and on_press_key:
        def dispatch(key):
            on_press_key(f'key.{key}')
        listener = TerminalKeyListener(dispatch)
        listener.start()
        logging.info("Using terminal keyboard input — keep this terminal focused.")
        return listener

    logging.warning(
        "Keyboard controls unavailable: no usable display (Wayland/headless) and stdin is "
        "not an interactive terminal.",
    )
    return None


def register_keyboard_controls(
    kb_state: KeyboardState = None,
    on_press: Optional[OnPressCallback] = None,
    on_release: Optional[OnReleaseCallback] = None,
    help_text: str = 'Keyboard Control'
) -> tuple[Optional[object], KeyboardState]:
    """Create a ``KeyboardState``, attach a listener, and return them.

    Args:
        kb_state: Initial keyboard state
        on_press: Optional callback ``(key_name, state) -> None`` invoked on
            every key press *after* the default handler and *before* the
            internal key_dict is updated.  Use this for script-specific
            events/actions (e.g. toggle pause, mark save).
        on_release: Optional callback ``(key_name, state) -> None`` invoked on
            every key release *after* the default handler and *before* the
            internal key_dict is updated.
        help_text: print after initial

    Returns:
        A tuple ``(listener, state)``.  ``listener`` is ``None`` when no
        keyboard backend is available (headless non-tty); callers must provide
        their own fallback in that case.
    """
    key_dict = {
        'key.esc': False,   # exit
        'key.enter': False, # help
    }
    events = {
        'exit': False,
    }
    if kb_state and isinstance(kb_state, KeyboardState):
        kb_state.key_dict.update(key_dict)
        kb_state.events.update(events)
    else:
        kb_state = KeyboardState(key_dict=key_dict, events=events)

    def _on_press(key: str) -> None:
        if kb_state.get_event("exit"):
            return
        if key not in kb_state.key_dict:
            return
        if key == "key.esc":
            print("\nEscape key pressed. Stopping ...")
            kb_state.set_event("exit")
        elif key == 'key.enter':
            kb_state.reset_keys()
        if on_press is not None:
            on_press(key)
        kb_state.key_dict[key] = True

    def _on_release(key: str) -> None:
        if kb_state.get_event("exit"):
            return
        if key not in kb_state.key_dict:
            return 
        if on_release:
            on_release(key)
        kb_state.key_dict[key] = False

    listener = _register_keyboard_listener(
        on_press_key=_on_press, on_release_key=_on_release
    )
    if listener is not None and on_press:
        if help_text:
            print(f'\n********** {help_text} Start **********')
        on_press('key.enter')
    return listener, kb_state
