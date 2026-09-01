import sys
import argparse
import logging
import time
from dataclasses import asdict
from pprint import pformat
import lerobot_robot_ufactory # patch
from lerobot.scripts.lerobot_record import (
    RecordConfig, RobotProcessorPipeline, LeRobotDataset, VideoEncodingManager,
    init_logging, make_robot_from_config, make_teleoperator_from_config, make_default_processors,
    combine_feature_dicts, aggregate_pipeline_dataset_features, create_initial_features,
    sanity_check_dataset_robot_compatibility, register_third_party_plugins
)
from lerobot.utils.visualization_utils import (
    init_visualization,
    shutdown_visualization,
)
from lerobot.configs import parser
from lerobot_robot_ufactory.scripts._uf_lerobot.record_loop import record_loop  # 覆盖星号导入的原生 record_loop
from lerobot_robot_ufactory.teleoperators.uf_mock_teleop import UFMockTeleop
from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop
from lerobot_robot_ufactory.utils.keyboard import KeyboardState, register_keyboard_controls


def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

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

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, 'cameras') else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split('/', 1)[-1]
            if repo_name.startswith('eval_'):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
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
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        # Connect the teleoperator before the robot so the robot isn't left idle (and possibly
        # tripping a firmware watchdog) during teleop init. Matches lerobot_record.py.
        if teleop is not None:
            teleop.connect()
        robot.connect()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.rgb_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        is_uf_teleop = isinstance(teleop, UFBaseTeleop)

        _events = {
            'exit': False,
            'exit_early': False,
            'rerecord_episode': False,
            'recorded': False,
        }
        _key_dict = {
            'key.esc': False,   # exit
            'key.enter': False, # help
            'key.space': False, # record
            'key.left': False,  # rerecord
            'key.right': False, # save
        }
        kb_state = KeyboardState(key_dict=_key_dict, events=_events)

        def on_key_press(key):
            if key == 'key.enter':
                if kb_state.get_event('recorded'):
                    print('⌨   [ESC] Exit  [←] Discard(Rerecord)  [→] Save')
                else:
                    print('⌨   [ESC] Exit  [Space] Start Record')
            elif key == 'key.esc':
                kb_state.set_event('exit_early', True)
            elif key == 'key.left':
                kb_state.set_event('rerecord_episode', True)
                kb_state.set_event('exit_early', True)
            elif key == 'key.right':
                kb_state.set_event('exit_early', True)
            elif key == 'key.space':
                if not kb_state.get_event('recorded'):
                    kb_state.set_event('recorded', True)

        listener, kb_state = register_keyboard_controls(
            kb_state=kb_state,
            on_press=on_key_press,
            help_text='Episode Record Loop',
        )
        if not listener:
            input('⌨   Press Enter to start record >>> ')
            if is_uf_teleop:
                teleop.set_teleop_enabled(True)
            kb_state.set_event('recorded', True)
            print('\n********** Episode Record Loop Start **********')

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while not kb_state.get_event('exit') and recorded_episodes < cfg.dataset.num_episodes:
                time.sleep(0.01)

                if teleop is not None and isinstance(teleop, UFMockTeleop):
                    if kb_state.get_event('exit'):
                        continue
                    teleop.configure(events=kb_state.events)
                    if kb_state.get_event('rerecord_episode'):
                        kb_state.set_event('rerecord_episode', False)
                        kb_state.set_event('exit_early', False)
                        input('\n⌨   Press Enter to regenerate random target location >>>>> ')
                        continue
                    if kb_state.get_event('exit'):
                        continue
                    kb_state.set_event('recorded', True)

                if kb_state.get_event('recorded'):
                    kb_state.set_event("rerecord_episode", False)
                    kb_state.set_event("exit_early", False)
                    if is_uf_teleop:
                        robot.configure()
                        obs = robot.get_observation()
                        teleop.set_teleop_enabled(True, obs)
                    print(f'Recording episode {dataset.num_episodes}')
                    if listener:
                        print('⌨   [ESC] Exit  [←] Discard(Rerecord)  [→] Save')
                    record_loop(
                        robot=robot,
                        events=kb_state.events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        dataset=dataset,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_mode=cfg.display_mode,
                        display_compressed_images=display_compressed_images,
                    )
                else:
                    continue
                if kb_state.get_event('exit'):
                    break
                if kb_state.get_event('rerecord_episode'):
                    print('Re-record episode')
                    kb_state.set_event('rerecord_episode', False)
                    kb_state.set_event('exit_early', False)
                    if is_uf_teleop:
                        teleop.set_teleop_enabled(False)
                    dataset.clear_episode_buffer()
                    kb_state.set_event('recorded', False)
                    if listener:
                        print('⌨   [ESC] Exit  [Space] Start Re-record')
                    else:
                        input('\n⌨   Press Enter to rerecord this episode >>>>> ')
                        kb_state.set_event('recorded', True)
                    continue

                if kb_state.get_event('recorded') and not kb_state.get_event('exit'):
                    if is_uf_teleop:
                        teleop.set_teleop_enabled(False)
                    print(f'[Episode {dataset.num_episodes}] saving ...')
                    dataset.save_episode()
                    print(f'[Episode {dataset.num_episodes}] save finish')
                    recorded_episodes += 1
                    kb_state.set_event('recorded', False)
                    if listener:
                        print('⌨   [ESC] Exit  [Space] Start Record')
                    else:
                        input('⌨   Press Enter to record at the next episode >>>>> ')
                        kb_state.set_event('recorded', True)
    finally:
        print('\n********** Episode Record Loop Exit **********')

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning('No episodes saved — skipping push to hub')
    return dataset

@parser.wrap()
def get_cfg(cfg: RecordConfig) -> RecordConfig:
    return cfg

def main():
    arg_parser = argparse.ArgumentParser(description='configuration args')
    arg_parser.add_argument('-r',
                       action='store_true', # specify --resume if resume needs to be True
                       default=False,
                       help='Whether contitue recording on existing dataset (default: False)')
    arg_parser.add_argument('-s', '--streaming-encoding',
                       action='store_true',
                       default=False,
                       help='Streaming encoding or not')
    args, unknown = arg_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    register_third_party_plugins()
    cfg = get_cfg()
    if args.r:
        cfg.resume = True
    cfg.play_sounds = False
    if args.streaming_encoding and not cfg.dataset.streaming_encoding:
        cfg.dataset.streaming_encoding = True
    record(cfg)


if __name__ == '__main__':
    main()
