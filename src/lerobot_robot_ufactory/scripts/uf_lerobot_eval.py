import sys
import argparse
import logging
import time
import math
from pathlib import Path
from dataclasses import asdict, dataclass
from pprint import pformat
import numpy as np
import lerobot_robot_ufactory # patch
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.policies.utils import make_robot_action
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.constants import OBS_STR
from lerobot.processor import (
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.common.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_ufactory.utils.keyboard import KeyboardState, register_keyboard_controls
from lerobot_robot_ufactory.utils.pose_utils import (
    POSE_KEYS,
    continuous_rotvec,
    blend_poses,
    compute_relative_axis_angle,
    compute_target_axis_angle,
    to_relative_pose,
    to_absolute_pose,
    get_pose_from_feature_dict,
    update_feature_dict_from_pose,
)
from lerobot_robot_ufactory.configs.parser import wrap as _uf_config_parser_wrap
parser.wrap = _uf_config_parser_wrap


@dataclass
class EvalConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: dict = None # no use
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    n_episodes: int = 50
    single_task: str | None = "pick_place"

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            policy_path = Path(policy_path).expanduser()
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Choose a policy to control the robot")
    
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def eval_loop(cfg: EvalConfig, relative=False, rx_continuous=False, camera_width=0, camera_height=0):
    init_logging()
    if camera_width or camera_height:
        cfg.robot.cameras_args = {'w': camera_width, 'h': camera_height}
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    try:
        dataset_metadata = LeRobotDatasetMetadata(repo_id=cfg.dataset.repo_id, root=cfg.dataset.root)
        dataset_features = dataset_metadata.features
        print("Loaded dataset metadata successfully.")
    except FileExistsError:
        # 预期路径：dataset 不存在，重新创建
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(
                    action=robot.action_features
                ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
                use_videos=cfg.dataset.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=cfg.dataset.video,
            ),
        )
        # Create empty dataset or load existing saved episodes
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        dataset_metadata = dataset.meta
        print("Created new dataset metadata successfully.")
    except Exception:
        logging.exception("Unexpected failure loading dataset metadata")
        raise
    
    if cfg.dataset.fps != dataset_metadata.fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset_metadata.fps} != {cfg.dataset.fps}).")
        

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset_metadata)
    # policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    # rename_map is loaded natively from the checkpoint's preprocessor_config.json (lerobot >= 0.6).
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
        dataset_stats=dataset_metadata.stats
    )

    robot.connect()

    device = get_safe_torch_device(policy.config.device, log=True)
    sleep_time_s = 1 / dataset_metadata.fps

    # Gripper look-ahead: denormalization stats for peeking into the action queue
    _gripper_mean = dataset_metadata.stats['action']['mean'][-1].item()
    _gripper_std = dataset_metadata.stats['action']['std'][-1].item()
    _gripper_min = dataset_metadata.stats['action']['min'][-1].item()
    _gripper_max = dataset_metadata.stats['action']['max'][-1].item()
    is_act_policy = hasattr(policy.config, 'chunk_size')
    # ACT: lookahead 30 (~1s) 补偿 chunk 慢启动; DP: 队列仅 8 步，lookahead 4
    GRIPPER_LOOKAHEAD = 0 if is_act_policy else 4

    # =====================================================
    # Chunk boundary smoothing: only damp large discontinuities at action chunk
    # boundaries (mean ~6mm jump) while preserving smooth within-chunk motion (~1mm).
    # When step-to-step cmd change exceeds SMOOTH_THRESHOLD, clamp it to that limit.
    # =====================================================
    SMOOTH_THRESHOLD = 0  # mm: smooth chunk boundary jumps while preserving trajectory
    SMOOTH_ROT_THRESHOLD = 0.05  # rad: max allowed rotation jump per step
    prev_smoothed_pose = None
    print(f'relative: {relative}')

    _events = {
        'exit': False,
        'reset': True,
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
                if not kb_state.get_event('reset'):
                    print('⌨   [ESC] Exit  [Space] Resume  [←] Stop(Reset)')
                else:
                    print('⌨   [ESC] Exit  [Space] Start')
            else:
                print('⌨   [ESC] Exit  [Space] Pause  [←] Stop(Reset)')
        elif key == 'key.space':
            kb_state.set_event('paused', not kb_state.get_event('paused'))
            if kb_state.get_event('paused'):
                if not kb_state.get_event('reset'):
                    print('⌨   [ESC] Exit  [Space] Resume  [←] Stop(Reset)')
                else:
                    print('⌨   [ESC] Exit  [Space] Start')
            else:
                print('⌨   [ESC] Exit  [Space] Pause  [←] Stop(Reset)')
        elif key == 'key.left':
            kb_state.set_event('reset', True)

    listener, kb_state = register_keyboard_controls(
        kb_state=kb_state,
        on_press=on_key_press,
        help_text='Policy Eval Episode Loop',
    )
    if not listener:
        print('\n********** Policy Eval Episode Loop Start **********')

    try:
        # with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        while not kb_state.get_event('exit'):
            time.sleep(0.1)
            if kb_state.get_event('paused'):
                continue
            print('\n********** Policy Eval Episode Start **********')
            kb_state.set_event('reset', False)

            robot.configure()
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            obs = robot.get_observation()

            prev_robot_dict = {}
            prev_action_dict = {}

            if hasattr(cfg.robot, 'robots'):
                keys = cfg.robot.robots.keys()
            else:
                keys = ['']
            for key in keys:
                prefix = f'{key}.' if key else ''
                is_tcp = all([f'{prefix}{item}' in obs for item in POSE_KEYS])
                if is_tcp:
                    pose = get_pose_from_feature_dict(obs, prefix=prefix)
                else:
                    pose = []
                prev_robot_dict[key] = {'type': 1 if is_tcp else 0, 'pose': np.array(pose)}
                prev_action_dict[key] = {'type': 1 if is_tcp else 0, 'pose': np.array(pose)}

            is_first = True
            
            start_loop_t = time.perf_counter()
            while not kb_state.get_event('exit'):
                sleep_duration = sleep_time_s - (time.perf_counter() - start_loop_t)
                precise_sleep(max(0.0, sleep_duration))
                start_loop_t = time.perf_counter()

                if kb_state.get_event('reset'):
                    kb_state.set_event('paused', True)
                    print('\n********** Policy Eval Episode Stop **********')
                    print('⌨   [ESC] Exit  [Space] Start')
                    break
                elif kb_state.get_event('paused'):
                    continue

                # Get robot observation
                obs = robot.get_observation()

                curr_robot_dict = {}
                curr_action_dict = {}
                for key in keys:
                    prefix = f'{key}.' if key else ''
                    is_tcp = all([f'{prefix}{item}' in obs for item in POSE_KEYS])

                    if not is_tcp:
                        curr_robot_dict[key] = {'type': 0, 'pose': []}
                        continue
                    if rx_continuous and not relative and f'{prefix}pose.rx' in obs and obs[f'{prefix}pose.rx'] < 0:
                        obs[f'{prefix}pose.rx'] += 2 * np.pi
                    curr_robot_pose = get_pose_from_feature_dict(obs, prefix=prefix)
                    if not is_first:
                        prev_robot_pose = prev_robot_dict[key]['pose']
                        # curr_robot_pose[3:6] = continuous_rotvec(curr_robot_pose[3:6], prev_robot_pose[3:6])
                    curr_robot_dict[key] = {'type': 1, 'pose': curr_robot_pose}

                    if relative:
                        if is_first:
                            delta = np.zeros(6, dtype=np.float64)
                        else:
                            delta = to_relative_pose(prev_robot_pose, curr_robot_pose)
                        update_feature_dict_from_pose(obs, delta, prefix=prefix)
                    prev_robot_dict[key] = {'type': 1, 'pose': curr_robot_pose}

                # Applies a pipeline to the raw robot observation, default is IdentityProcessor
                obs_processed = robot_observation_processor(obs)
                observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

                action_values = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=cfg.single_task,
                    robot_type=robot.robot_type,
                )
                act_processed_policy = make_robot_action(action_values, dataset_features)
                robot_action_to_send = robot_action_processor((act_processed_policy, obs))

                for key in keys:
                    prefix = f'{key}.' if key else ''
                    if curr_robot_dict[key]['type'] != 1:
                        continue
                    if relative:
                        delta = get_pose_from_feature_dict(robot_action_to_send, prefix=prefix)
                        curr_robot_pose = curr_robot_dict[key]['pose']
                        target_action_pose = to_absolute_pose(delta, curr_robot_pose)
                        if not is_first:
                            prev_action_pose = prev_action_dict[key]['pose']
                            target_action_pose[3:6] = continuous_rotvec(target_action_pose[3:6], prev_action_pose[3:6])
                        update_feature_dict_from_pose(robot_action_to_send, target_action_pose, prefix=prefix)
                    elif rx_continuous and f'{prefix}pose.rx' in robot_action_to_send and robot_action_to_send[f'{prefix}pose.rx'] > math.pi:
                        robot_action_to_send[f'{prefix}pose.rx'] -= 2 * np.pi

                    # robot_action_to_send[f'{prefix}pose.z'] = max(robot_action_to_send[f'{prefix}pose.z'], 199)

                    # Rate-limited smoothing: cap position velocity to reduce chunk boundary jerks
                    # Uses vector-norm clamping to preserve motion direction
                    if SMOOTH_THRESHOLD > 0:
                        pos_keys = [f'{prefix}{key}' for key in POSE_KEYS[:3]]
                        rot_keys = [f'{prefix}{key}' for key in POSE_KEYS[3:6]]
                        if prev_smoothed_pose is None:
                            prev_smoothed_pose = {k: robot_action_to_send[k] for k in pos_keys + rot_keys}
                        else:
                            # Vector-norm clamp on position (preserves direction)
                            delta_pos = np.array([robot_action_to_send[k] - prev_smoothed_pose[k] for k in pos_keys])
                            norm = np.linalg.norm(delta_pos)
                            if norm > SMOOTH_THRESHOLD:
                                delta_pos = delta_pos * (SMOOTH_THRESHOLD / norm)
                            for i, k in enumerate(pos_keys):
                                prev_smoothed_pose[k] = prev_smoothed_pose[k] + delta_pos[i]
                                robot_action_to_send[k] = prev_smoothed_pose[k]
                            # Per-axis clamp on rotation
                            for k in rot_keys:
                                delta = robot_action_to_send[k] - prev_smoothed_pose[k]
                                if abs(delta) > SMOOTH_ROT_THRESHOLD:
                                    delta = SMOOTH_ROT_THRESHOLD * (1 if delta > 0 else -1)
                                prev_smoothed_pose[k] = prev_smoothed_pose[k] + delta
                                robot_action_to_send[k] = prev_smoothed_pose[k]

                    if relative:
                        curr_action_pose = get_pose_from_feature_dict(robot_action_to_send, prefix=prefix)
                        # 相对增量模式: 漂移修正，将指令位姿温和拉回实际位姿
                        # prev_action_pose = blend_poses(curr_action_pose, curr_robot_pose, 0.05)
                        prev_action_dict[key]['pose'] = curr_action_pose

                # # Gripper look-ahead: peek ahead in the action queue to compensate
                # # for the slow ramp in the ACT chunk (eliminates 1-2s gripper delay)
                # gripper_raw = robot_action_to_send.get('left.gripper.pos', 0)
                # if hasattr(policy, '_action_queue') and len(policy._action_queue) > 0:
                #     # ACT: 队列为 deque of tensors, 归一化方式 MEAN_STD
                #     lookahead_idx = min(GRIPPER_LOOKAHEAD, len(policy._action_queue) - 1)
                #     future_gripper_norm = policy._action_queue[lookahead_idx][0, -1].item()
                #     gripper_raw = future_gripper_norm * _gripper_std + _gripper_mean
                # elif hasattr(policy, '_queues') and 'action' in policy._queues and len(policy._queues['action']) > 0:
                #     # DP: 队列结构不同, 归一化方式 MIN_MAX → [-1,1] → [min,max]
                #     lookahead_idx = min(GRIPPER_LOOKAHEAD, len(policy._queues['action']) - 1)
                #     future_gripper_norm = policy._queues['action'][lookahead_idx][0, -1].item()
                #     gripper_raw = (future_gripper_norm + 1) / 2 * (_gripper_max - _gripper_min) + _gripper_min
                # robot_action_to_send['left.gripper.pos'] = 1.0 if gripper_raw > 0.4 else 0.0

                # gripper_raw = robot_action_to_send.get('right.gripper.pos', 0)
                # if hasattr(policy, '_action_queue') and len(policy._action_queue) > 0:
                #     # ACT: 队列为 deque of tensors, 归一化方式 MEAN_STD
                #     lookahead_idx = min(GRIPPER_LOOKAHEAD, len(policy._action_queue) - 1)
                #     future_gripper_norm = policy._action_queue[lookahead_idx][0, -1].item()
                #     gripper_raw = future_gripper_norm * _gripper_std + _gripper_mean
                # elif hasattr(policy, '_queues') and 'action' in policy._queues and len(policy._queues['action']) > 0:
                #     # DP: 队列结构不同, 归一化方式 MIN_MAX → [-1,1] → [min,max]
                #     lookahead_idx = min(GRIPPER_LOOKAHEAD, len(policy._queues['action']) - 1)
                #     future_gripper_norm = policy._queues['action'][lookahead_idx][0, -1].item()
                #     gripper_raw = (future_gripper_norm + 1) / 2 * (_gripper_max - _gripper_min) + _gripper_min
                # robot_action_to_send['right.gripper.pos'] = 1.0 if gripper_raw > 0.4 else 0.0

                is_first = False
                robot.send_action(robot_action_to_send)
    finally:
        print('\n********** Policy Eval Loop Exit **********')
        robot.disconnect()
        if listener is not None:
            listener.stop()

@parser.wrap()
def get_cfg(cfg: EvalConfig) -> EvalConfig:
    return cfg

def main():
    arg_parser = argparse.ArgumentParser(description='configuration args')
    arg_parser.add_argument('--relative', action='store_true', help='is relative motion or not')
    arg_parser.add_argument('--rx_continuous', action='store_true', help='rx continuous or not')
    arg_parser.add_argument('--camera_width', type=int, required=False, default=0)
    arg_parser.add_argument('--camera_height', type=int, required=False, default=0)
    args, unknown = arg_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    register_third_party_plugins()
    cfg = get_cfg()
    eval_loop(cfg, args.relative, args.rx_continuous, camera_width=args.camera_width, camera_height=args.camera_height)


if __name__ == "__main__":
    main()
