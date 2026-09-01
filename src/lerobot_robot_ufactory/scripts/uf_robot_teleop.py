import sys
import argparse
import logging
import time
from pathlib import Path
from dataclasses import asdict, dataclass
from pprint import pformat
import lerobot_robot_ufactory # patch
from lerobot.processor import (
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
)
from lerobot_robot_ufactory.configs import parser
from lerobot_robot_ufactory.utils.keyboard import KeyboardState, register_keyboard_controls
from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop


@dataclass
class TeleopConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    fps: int = 30

    def __post_init__(self):
        if hasattr(self.robot, 'robots'):
            for _, robot in self.robot.robots.items():
                robot.cameras = {}
        else:
            self.robot.cameras = {}


def teleop_loop(cfg: TeleopConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)
    if hasattr(cfg.robot, "teleop"):
        cfg.robot.teleop = teleop
    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    try:
        robot.connect()
        teleop.connect()

        sleep_time_s = 1 / cfg.fps

        is_uf_teleop = isinstance(teleop, UFBaseTeleop)

        _events = {
            'exit': False,
            'reset': False,
            'paused': True
        }
        _key_dict = {
            'key.esc': False,   # exit
            'key.enter': False, # help
            'key.space': False, # pause/start
            'key.left': False,  # reset
        }
        kb_state = KeyboardState(key_dict=_key_dict, events=_events)

        def on_key_press(key):
            if key == 'key.enter':
                if kb_state.get_event('paused'):
                    if kb_state.get_event('reset'):
                        print('⌨   [ESC] Exit  [Space] Reset / Start  [←] Reset')
                    else:
                        print('⌨   [ESC] Exit  [Space] Start  [←] Reset')
                else:
                    print('⌨   [ESC] Exit  [Space] Pause  [←] Pause / Reset')
            elif key == 'key.left':
                kb_state.set_event('reset')

        listener, kb_state = register_keyboard_controls(
            kb_state=kb_state,
            on_press=on_key_press,
            help_text='Teleop Control Loop',
        )
        if not listener:
            input('⌨   Press Enter to start teleop >>> ')
            if is_uf_teleop:
                obs = robot.get_observation()
                teleop.set_teleop_enabled(True, obs)
            kb_state.set_event('paused', False)
            kb_state.set_event('reset', False)
            print('\n********** Teleop Control Loop Start **********')

        key_space_pressed = False
        key_left_pressed = False

        start_loop_t = time.perf_counter()

        while not kb_state.is_pressed('key.esc'):
            sleep_duration = sleep_time_s - (time.perf_counter() - start_loop_t)
            precise_sleep(max(0.0, sleep_duration))
            start_loop_t = time.perf_counter()

            if listener:
                if kb_state.is_pressed('key.left') and not key_left_pressed:
                    key_left_pressed = True
                    kb_state.set_event('reset', True)
                    if not kb_state.get_event('paused'):
                        kb_state.set_event('paused', True)
                        if is_uf_teleop:
                            teleop.set_teleop_enabled(False)
                    print('⌨   [ESC] Exit  [Space] Reset / Start  [←] Reset')
                elif not kb_state.is_pressed('key.left') and key_left_pressed:
                    key_left_pressed = False

                if kb_state.is_pressed('key.space') and not key_space_pressed:
                    key_space_pressed = True
                    kb_state.set_event('paused', not kb_state.get_event('paused'))
                    if kb_state.get_event('paused'):
                        if is_uf_teleop:
                            teleop.set_teleop_enabled(False)
                        # # print('========== Teleop is paused ==========')
                        print('⌨   [ESC] Exit  [Space] Start  [←] Reset')
                    else:
                        if kb_state.get_event('reset'):
                            kb_state.set_event('reset', False)
                            robot.configure()
                        # print('========== Teleop is start ==========')
                        if is_uf_teleop:
                            obs = robot.get_observation()
                            teleop.set_teleop_enabled(True, obs)
                        print('⌨   [ESC] Exit  [Space] Pause  [←] Reset')
                    continue
                elif not kb_state.is_pressed('key.space') and key_space_pressed:
                    key_space_pressed = False

                if kb_state.get_event('reset') or kb_state.get_event('paused'):
                    continue

            # Get robot observation
            obs = robot.get_observation()

            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs))

            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
            robot.send_action(robot_action_to_send)
    finally:
        print('\n********** Teleop Control Loop Exit **********')
        robot.disconnect()
        teleop.disconnect()
        if listener is not None:
            listener.stop()

@parser.wrap()
def get_cfg(cfg: TeleopConfig) -> TeleopConfig:
    return cfg

def main():
    arg_parser = argparse.ArgumentParser(description='configuration args')
    args, unknown = arg_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    register_third_party_plugins()
    cfg = get_cfg()
    teleop_loop(cfg)


if __name__ == "__main__":
    main()
