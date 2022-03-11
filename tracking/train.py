#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))
import numpy as np
import ray
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from typing import Dict
import multiprocessing
from collections import defaultdict
import logging
import klimits


METRIC_OPS = ['sum', 'average', 'max', 'min']


class CustomTrainCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):

        episode.user_data['op'] = {}
        for op in METRIC_OPS:
            episode.user_data['op'][op] = defaultdict(list)

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):

        episode_info = episode.last_info_for()
        if episode_info:
            for op in list(episode_info.keys() & METRIC_OPS):
                for k, v in episode_info[op].items():
                    episode.user_data['op'][op][k].append(v)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        def __apply_op_on_list(operator, data_list):
            if operator == 'sum':
                return sum(data_list)
            elif operator == 'average':
                return sum(data_list) / len(data_list)
            elif operator == 'max':
                return max(data_list)
            elif operator == 'min':
                return min(data_list)

        episode_info = episode.last_info_for()
        episode_length = episode_info['episode_length']
        trajectory_length = episode_info['trajectory_length']
        env = base_env.get_unwrapped()[-1]

        for op in METRIC_OPS:
            for k, v in episode.user_data['op'][op].items():
                episode.custom_metrics[k + '_' + op] = __apply_op_on_list(op, episode.user_data['op'][op][k])

        for k, v in episode_info.items():
            if k.startswith('obstacles'):
                episode.custom_metrics[k] = v
            if "spline" in k and not np.isnan(v):
                episode.custom_metrics[k] = v

        episode.custom_metrics['episode_length'] = float(episode_length)
        episode.custom_metrics['trajectory_length'] = trajectory_length

        if 'trajectory_fraction' in episode_info:
            episode.custom_metrics['trajectory_fraction'] = episode_info['trajectory_fraction']

        # if 'reference_spline_length' in episode_info:  # should be covered by spline in k
        #    episode.custom_metrics['reference_spline_length'] = episode_info['reference_spline_length']

        if 'trajectory_successful' in episode_info:
            episode.custom_metrics['success_rate'] = episode_info['trajectory_successful']
        else:
            episode.custom_metrics['success_rate'] = 0.0

        if episode_info['termination_reason'] == env.TERMINATION_JOINT_LIMITS:
            episode.custom_metrics['joint_limit_violation_termination_rate'] = 1.0
        else:
            episode.custom_metrics['joint_limit_violation_termination_rate'] = 0.0

        if env.use_splines:
            termination_reasons_spline_dict = {env.TERMINATION_TRAJECTORY_LENGTH: 'trajectory_length_termination_rate',
                                               env.TERMINATION_SPLINE_LENGTH: 'spline_length_termination_rate',
                                               env.TERMINATION_SPLINE_DEVIATION: 'spline_deviation_termination_rate',
                                               env.TERMINATION_ROBOT_STOPPED: 'robot_stopped_termination_rate'}

            for k, v in termination_reasons_spline_dict.items():
                episode.custom_metrics[v] = 1.0 if episode_info['termination_reason'] == k else 0.0

            if env.sphere_balancing_mode:
                if episode_info['termination_reason'] == env.TERMINATION_BALANCING:
                    episode.custom_metrics['balancing_termination_rate'] = 1.0
                else:
                    episode.custom_metrics['balancing_termination_rate'] = 0.0

        if env.floating_robot_base:
            termination_reasons_base_dict = \
                {env.TERMINATION_BASE_POS_DEVIATION: 'robot_base_pos_deviation_termination_rate',
                 env.TERMINATION_BASE_ORN_DEVIATION: 'robot_base_orn_deviation_termination_rate',
                 env.TERMINATION_BASE_Z_ANGLE_DEVIATION: 'robot_base_z_angle_deviation_termination_rate'}

            for k, v in termination_reasons_base_dict.items():
                episode.custom_metrics[v] = 1.0 if episode_info['termination_reason'] == k else 0.0

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result['callback_ok'] = True


def _make_env_config():

    env_config = {
        'experiment_name': args.name,
        'm_prev': args.m_prev,
        'pos_limit_factor': args.pos_limit_factor,
        'vel_limit_factor': args.vel_limit_factor,
        'acc_limit_factor': args.acc_limit_factor,
        'jerk_limit_factor': args.jerk_limit_factor,
        'torque_limit_factor': args.torque_limit_factor,
        'action_mapping_factor': args.action_mapping_factor,
        'set_highest_action_to_one': args.set_highest_action_to_one,
        'action_preprocessing_function': args.action_preprocessing_function,
        'normalize_reward_to_frequency': args.normalize_reward_to_frequency,
        'online_trajectory_duration': args.online_trajectory_duration,
        'online_trajectory_time_step': args.online_trajectory_time_step,
        'obs_add_target_point_pos': args.obs_add_target_point_pos,
        'obs_add_target_point_relative_pos': args.obs_add_target_point_relative_pos,
        'punish_action': args.punish_action,
        'action_punishment_min_threshold': args.action_punishment_min_threshold,
        'action_max_punishment': args.action_max_punishment,
        'reward_action': args.reward_action,
        'action_reward_min_threshold': args.action_reward_min_threshold,
        'action_reward_peak': args.action_reward_peak,
        'action_max_reward': args.action_max_reward,
        'punish_adaptation': args.punish_adaptation,
        'adaptation_max_punishment': args.adaptation_max_punishment,
        'punish_end_min_distance': args.punish_end_min_distance,
        'end_min_distance_max_threshold': args.end_min_distance_max_threshold,
        'end_min_distance_max_punishment': args.end_min_distance_max_punishment,
        'punish_end_max_torque': args.punish_end_max_torque,
        'end_max_torque_min_threshold': args.end_max_torque_min_threshold,
        'end_max_torque_max_punishment': args.end_max_torque_max_punishment,
        'obstacle_scene': args.obstacle_scene,
        'activate_obstacle_collisions': args.activate_obstacle_collisions,
        'log_obstacle_data': False,
        'check_braking_trajectory_collisions': args.check_braking_trajectory_collisions,
        'check_braking_trajectory_torque_limits': args.check_braking_trajectory_torque_limits,
        'collision_check_time': args.collision_check_time,
        'closest_point_safety_distance': args.closest_point_safety_distance,
        'use_target_points': args.use_target_points,
        'acc_limit_factor_braking': args.acc_limit_factor_braking,
        'jerk_limit_factor_braking': args.jerk_limit_factor_braking,
        'punish_braking_trajectory_min_distance': args.punish_braking_trajectory_min_distance,
        'braking_trajectory_min_distance_max_threshold': args.braking_trajectory_min_distance_max_threshold,
        'braking_trajectory_max_punishment': args.braking_trajectory_max_punishment,
        'punish_braking_trajectory_max_torque': args.punish_braking_trajectory_max_torque,
        'braking_trajectory_max_torque_min_threshold': args.braking_trajectory_max_torque_min_threshold,
        'robot_scene': args.robot_scene,
        'num_virtual_motors': args.num_virtual_motors,
        'no_self_collision': args.no_self_collision,
        'terminate_on_robot_stop': args.terminate_on_robot_stop,
        'use_controller_target_velocities': args.use_controller_target_velocities,
        'target_point_cartesian_range_scene': args.target_point_cartesian_range_scene,
        'target_point_relative_pos_scene': args.target_point_relative_pos_scene,
        'target_point_radius': args.target_point_radius,
        'target_point_sequence': args.target_point_sequence,
        'target_point_reached_reward_bonus': args.target_point_reached_reward_bonus,
        'target_point_reward_factor': args.target_point_reward_factor,
        'target_point_use_actual_position': args.target_point_use_actual_position,
        'normalize_reward_to_initial_target_point_distance': args.normalize_reward_to_initial_target_point_distance,
        'use_splines': args.spline_dir is not None,
        'spline_dir': args.spline_dir,
        'spline_config_path': args.spline_config_path,
        'spline_u_arc_start_range': args.spline_u_arc_start_range,
        'spline_u_arc_diff_min': args.spline_u_arc_diff_min,
        'spline_deviation_weighting_factors': args.spline_deviation_weighting_factors,
        'spline_speed_range': args.spline_speed_range,
        'spline_random_speed_per_time_step': args.spline_random_speed_per_time_step,
        'spline_normalize_duration': args.spline_normalize_duration,
        'spline_use_reflection_vectors': args.spline_use_reflection_vectors,
        'spline_final_overshoot_factor': args.spline_final_overshoot_factor,
        'spline_final_distance_reward': args.spline_final_distance_reward,
        'spline_termination_max_deviation': args.spline_termination_max_deviation,
        'spline_termination_extra_time_steps': args.spline_termination_extra_time_steps,
        'spline_max_final_deviation': args.spline_max_final_deviation,
        'spline_braking_extra_time_steps': args.spline_braking_extra_time_steps,
        'obs_spline_n_next': args.obs_spline_n_next,
        'obs_spline_add_length': args.obs_spline_add_length,
        'obs_spline_add_distance_per_knot': args.obs_spline_add_distance_per_knot,
        'obs_spline_use_distance_between_knots': args.obs_spline_use_distance_between_knots,
        'spline_distance_max_reward': args.spline_distance_max_reward,
        'spline_deviation_max_threshold': args.spline_deviation_max_threshold,
        'punish_spline_max_deviation': args.punish_spline_max_deviation,
        'spline_max_deviation_max_punishment': args.spline_max_deviation_max_punishment,
        'punish_spline_mean_deviation': args.punish_spline_mean_deviation,
        'spline_mean_deviation_max_punishment': args.spline_mean_deviation_max_punishment,
        'spline_cartesian_deviation_max_threshold': args.spline_cartesian_deviation_max_threshold,
        'punish_spline_max_cartesian_deviation': args.punish_spline_max_cartesian_deviation,
        'spline_max_cartesian_deviation_max_punishment': args.spline_max_cartesian_deviation_max_punishment,
        'punish_spline_mean_cartesian_deviation': args.punish_spline_mean_cartesian_deviation,
        'spline_mean_cartesian_deviation_max_punishment': args.spline_mean_cartesian_deviation_max_punishment,
        'sphere_balancing_mode': args.sphere_balancing_mode,
        'balancing_sphere_dev_min_max': args.balancing_sphere_dev_min_max,
        'terminate_on_balancing_sphere_deviation': args.terminate_on_balancing_sphere_deviation,
        'terminate_balancing_sphere_not_on_board': args.terminate_balancing_sphere_not_on_board,
        'balancing_sphere_max_reward': args.balancing_sphere_max_reward,
        'floating_robot_base': args.floating_robot_base,
        'robot_base_balancing_mode': args.robot_base_balancing_mode,
        'balancing_robot_base_max_pos_deviation': args.balancing_robot_base_max_pos_deviation,
        'balancing_robot_base_pos_max_reward': args.balancing_robot_base_pos_max_reward,
        'balancing_robot_base_max_orn_deviation': args.balancing_robot_base_max_orn_deviation,
        'balancing_robot_base_orn_max_reward': args.balancing_robot_base_orn_max_reward,
        'balancing_robot_base_max_z_angle_deviation_rad': args.balancing_robot_base_max_z_angle_deviation_rad,
        'balancing_robot_base_z_angle_max_reward': args.balancing_robot_base_z_angle_max_reward,
        'balancing_robot_base_punish_last_cartesian_action_point':
            args.balancing_robot_base_punish_last_cartesian_action_point,
        'balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold':
            args.balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold,
        'balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment':
            args.balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment,
        'terminate_on_balancing_robot_base_pos_deviation': args.terminate_on_balancing_robot_base_pos_deviation,
        'terminate_on_balancing_robot_base_orn_deviation': args.terminate_on_balancing_robot_base_orn_deviation,
        'terminate_on_balancing_robot_base_z_angle_deviation': args.terminate_on_balancing_robot_base_z_angle_deviation,
        'target_link_offset': args.target_link_offset,
        'obstacle_use_computed_actual_values': args.obstacle_use_computed_actual_values,
        'solver_iterations': args.solver_iterations,
        'logging_level': args.logging_level,
    }

    if hasattr(klimits, '__version__'):
        env_config['klimits_version'] = klimits.__version__

    if hasattr(ray, '__version__'):
        env_config['ray_version'] = ray.__version__

    return env_config


if __name__ == '__main__':
    config = {
        'model': {
            'conv_filters': None,
            'fcnet_hiddens': [256, 128],
            'fcnet_activation': None,  # set at a later point
            'use_lstm': False,
        },
        'gamma': 0.99,
        'use_gae': True,
        'lambda': 1.0,
        'kl_coeff': 0.2,
        'rollout_fragment_length': None,  # set at a later point
        'train_batch_size': 49920,
        'sgd_minibatch_size': 1024,
        'num_sgd_iter': 16,
        'lr': 5e-5,
        'lr_schedule': None,
        'vf_loss_coeff': 1.0,
        'entropy_coeff': None,
        'clip_param': 0.3,
        'vf_clip_param': None,
        'kl_target': None,
        'batch_mode': 'complete_episodes',
        'normalize_actions': True,
        'evaluation_interval': None,
        'evaluation_num_episodes': 624,
        'evaluation_parallel_to_training': False,
        'evaluation_config': {
            "explore": False,
            "rollout_fragment_length": 1},
        # sample a single episode per self.evaluation_workers.local_worker().sample() in trainer.py
        'evaluation_num_workers': 0,
    }

    algorithm = 'PPO'
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default_name')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint if training should be continued.')
    parser.add_argument('--time', type=int, required=True,
                        help='Total time of the training in hours.')
    parser.add_argument('--iterations_per_checkpoint', type=int, default=500,
                        help='The number of training iterations per checkpoint')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--num_threads_per_worker', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--m_prev', type=int, default=0)
    parser.add_argument('--pos_limit_factor', type=float, default=1.0)
    parser.add_argument('--vel_limit_factor', type=float, default=1.0)
    parser.add_argument('--acc_limit_factor', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor', type=float, default=1.0)
    parser.add_argument('--torque_limit_factor', type=float, default=1.0)
    parser.add_argument('--action_mapping_factor', type=float, default=1.0)
    parser.add_argument('--set_highest_action_to_one', action='store_true', default=False)
    parser.add_argument('--action_preprocessing_function', default=None, choices=['tanh'])
    parser.add_argument('--normalize_reward_to_frequency', dest='normalize_reward_to_frequency', action='store_true',
                        default=False)
    parser.add_argument('--batch_size_factor', type=float, default=1.0)
    parser.add_argument('--online_trajectory_duration', type=float, default=8.0)
    parser.add_argument('--online_trajectory_time_step', type=float, default=0.1)
    parser.add_argument('--obs_add_target_point_pos', action='store_true', default=False)
    parser.add_argument('--obs_add_target_point_relative_pos', action='store_true', default=False)
    parser.add_argument('--punish_action', action='store_true', default=False)
    parser.add_argument('--action_punishment_min_threshold', type=float, default=0.9)
    parser.add_argument('--action_max_punishment', type=float, default=0.5)
    parser.add_argument('--reward_action', action='store_true', default=False)
    parser.add_argument('--action_reward_min_threshold', type=float, default=0.9)
    parser.add_argument('--action_reward_peak', type=float, default=0.95)
    parser.add_argument('--action_max_reward', type=float, default=1.0)
    parser.add_argument('--punish_adaptation', action='store_true', default=False)
    parser.add_argument('--adaptation_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_end_min_distance', action='store_true', default=False)
    parser.add_argument('--end_min_distance_max_threshold', type=float, default=0.05)
    parser.add_argument('--end_min_distance_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_end_max_torque', action='store_true', default=False)
    parser.add_argument('--end_max_torque_min_threshold', type=float, default=0.9)
    parser.add_argument('--end_max_torque_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_braking_trajectory_min_distance', action='store_true', default=False)
    parser.add_argument('--braking_trajectory_min_distance_max_threshold', type=float, default=0.05)
    parser.add_argument('--braking_trajectory_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_braking_trajectory_max_torque', action='store_true', default=False)
    parser.add_argument('--braking_trajectory_max_torque_min_threshold', type=float, default=0.8)
    parser.add_argument('--obstacle_scene', type=int, default=0)
    parser.add_argument('--activate_obstacle_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_torque_limits', action='store_true', default=False)
    parser.add_argument('--collision_check_time', type=float, default=None)
    parser.add_argument('--closest_point_safety_distance', type=float, default=0.1)
    parser.add_argument('--use_target_points', action='store_true', default=False)
    parser.add_argument('--acc_limit_factor_braking', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor_braking', type=float, default=1.0)
    parser.add_argument('--robot_scene', type=int, default=0)
    parser.add_argument('--num_virtual_motors', type=int, default=0)
    parser.add_argument('--no_self_collision', action='store_true', default=False)
    parser.add_argument('--terminate_on_robot_stop', action='store_true', default=False)
    parser.add_argument('--use_controller_target_velocities', action='store_true', default=False)
    parser.add_argument('--target_point_cartesian_range_scene', type=int, default=0)
    parser.add_argument('--target_point_relative_pos_scene', type=int, default=0)
    parser.add_argument('--target_point_radius', type=float, default=0.065)
    parser.add_argument('--target_point_sequence', type=int, default=0)
    parser.add_argument('--target_point_reached_reward_bonus', type=float, default=0.00)
    parser.add_argument('--target_point_use_actual_position', action='store_true', default=False)
    parser.add_argument('--target_link_offset', type=json.loads, default='[0, 0, 0.126]')
    parser.add_argument('--target_point_reward_factor', type=float, default=1.0)
    parser.add_argument('--normalize_reward_to_initial_target_point_distance', action='store_true', default=False)
    # spline settings
    parser.add_argument('--spline_dir', type=str, default=None)
    parser.add_argument('--spline_config_path', type=str, default=None)
    parser.add_argument('--spline_u_arc_start_range', type=json.loads, default=(0, 0))
    parser.add_argument('--spline_u_arc_diff_min', type=float, default=1.0)
    parser.add_argument('--spline_deviation_weighting_factors', type=json.loads, default=None)
    parser.add_argument('--spline_normalize_duration', action='store_true', default=False)
    parser.add_argument('--spline_use_reflection_vectors', action='store_true', default=False)
    parser.add_argument('--spline_speed_range', type=json.loads, default=None)
    parser.add_argument('--spline_random_speed_per_time_step', action='store_true', default=False)
    parser.add_argument('--spline_final_overshoot_factor', type=float, default=1.0)
    parser.add_argument('--spline_final_distance_reward', type=float, default=None)
    parser.add_argument('--spline_termination_max_deviation', type=float, default=None)
    parser.add_argument('--spline_termination_extra_time_steps', type=int, default=None)
    parser.add_argument('--spline_max_final_deviation', type=float, default=0.05)
    parser.add_argument('--spline_braking_extra_time_steps', type=float, default=None)
    parser.add_argument('--obs_spline_n_next', type=int, default=5)
    parser.add_argument('--obs_spline_add_length', action='store_true', default=False)
    parser.add_argument('--obs_spline_add_distance_per_knot', action='store_true', default=False)
    parser.add_argument('--obs_spline_use_distance_between_knots', action='store_true', default=False)
    parser.add_argument('--spline_distance_max_reward', type=float, default=1.0)
    parser.add_argument('--spline_deviation_max_threshold', type=float, default=0.25)
    parser.add_argument('--punish_spline_max_deviation', action='store_true', default=False)
    parser.add_argument('--spline_max_deviation_max_punishment', type=float, default=0.1)
    parser.add_argument('--punish_spline_mean_deviation', action='store_true', default=False)
    parser.add_argument('--spline_mean_deviation_max_punishment', type=float, default=0.1)
    parser.add_argument('--spline_cartesian_deviation_max_threshold', type=float, default=0.1)
    parser.add_argument('--punish_spline_max_cartesian_deviation', action='store_true', default=False)
    parser.add_argument('--spline_max_cartesian_deviation_max_punishment', type=float, default=0.1)
    parser.add_argument('--punish_spline_mean_cartesian_deviation', action='store_true', default=False)
    parser.add_argument('--spline_mean_cartesian_deviation_max_punishment', type=float, default=0.1)
    # end of spline settings
    # sphere balancing settings
    parser.add_argument('--sphere_balancing_mode', action='store_true', default=False)
    parser.add_argument('--balancing_sphere_dev_min_max', type=json.loads, default=None)
    parser.add_argument('--terminate_on_balancing_sphere_deviation', action='store_true', default=False)
    parser.add_argument('--terminate_balancing_sphere_not_on_board', action='store_true', default=False)
    parser.add_argument('--balancing_sphere_max_reward', type=float, default=1.0)
    # end of sphere balancing settings
    # robot base balancing settings
    parser.add_argument('--floating_robot_base', action='store_true', default=False)
    parser.add_argument('--robot_base_balancing_mode', action='store_true', default=False)
    parser.add_argument('--balancing_robot_base_max_pos_deviation', type=float, default=0.2)
    parser.add_argument('--balancing_robot_base_pos_max_reward', type=float, default=0.0)
    parser.add_argument('--balancing_robot_base_max_orn_deviation', type=float, default=1.0)
    parser.add_argument('--balancing_robot_base_orn_max_reward', type=float, default=0.0)
    parser.add_argument('--balancing_robot_base_max_z_angle_deviation_rad', type=float, default=0.52)
    parser.add_argument('--balancing_robot_base_z_angle_max_reward', type=float, default=0.0)
    parser.add_argument('--balancing_robot_base_punish_last_cartesian_action_point', action='store_true', default=False)
    parser.add_argument('--balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold',
                        type=float, default=0.4)
    parser.add_argument('--balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment',
                        type=float, default=1.0)
    parser.add_argument('--terminate_on_balancing_robot_base_pos_deviation', action='store_true', default=False)
    parser.add_argument('--terminate_on_balancing_robot_base_orn_deviation', action='store_true', default=False)
    parser.add_argument('--terminate_on_balancing_robot_base_z_angle_deviation', action='store_true', default=False)
    # end of robot base balancing settings
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--logging_level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--hidden_layer_activation', default='selu', choices=['relu', 'selu', 'tanh', 'sigmoid', 'elu',
                                                                              'gelu', 'swish', 'leaky_relu'])
    parser.add_argument('--last_layer_activation', default=None, choices=['linear', 'tanh'])
    parser.add_argument('--no_log_std_activation', action='store_true', default=False)
    parser.add_argument('--solver_iterations', type=int, default=None)
    parser.add_argument('--use_dashboard', action='store_true', default=False)
    parser.add_argument('--evaluation_interval', type=int, default=None)
    parser.add_argument('--vf_clip_param', type=float, default=10.0)
    parser.add_argument('--entropy_coeff', type=float, default=0.0)
    parser.add_argument('--kl_target', type=float, default=0.01)
    parser.add_argument('--fcnet_hiddens', type=json.loads, default=None)
    parser.add_argument('--log_std_range', type=json.loads, default=None)
    parser.add_argument('--action_distribution', default=None,
                        choices=['truncated_normal', 'truncated_normal_zero_kl', 'beta_alpha_beta'])
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(args.logging_level)

    config['model']['fcnet_activation'] = args.hidden_layer_activation
    if args.fcnet_hiddens is not None:
        config['model']['fcnet_hiddens'] = args.fcnet_hiddens
    config['evaluation_interval'] = args.evaluation_interval
    config['vf_clip_param'] = args.vf_clip_param
    config['entropy_coeff'] = args.entropy_coeff
    config['kl_target'] = args.kl_target

    if args.last_layer_activation is not None and args.last_layer_activation != 'linear':
        use_keras_model = True
        if use_keras_model:
            from tracking.model.keras_fcnet_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('keras_fcnet_last_layer_activation',
                                               FullyConnectedNetworkLastLayerActivation)
            config['model']['custom_model'] = 'keras_fcnet_last_layer_activation'
        else:
            from tracking.model.fcnet_v2_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('fcnet_last_layer_activation', FullyConnectedNetworkLastLayerActivation)
            config['model']['custom_model'] = 'fcnet_last_layer_activation'

        config['model']['custom_model_config'] = {'last_layer_activation': args.last_layer_activation,
                                                  'no_log_std_activation': args.no_log_std_activation,
                                                  'log_std_range': args.log_std_range}
        if use_keras_model:
            for key in ['fcnet_hiddens', 'fcnet_activation', 'post_fcnet_hiddens', 'post_fcnet_activation',
                        'no_final_layer', 'vf_share_layers', 'free_log_std']:
                if key in config['model']:
                    config['model']['custom_model_config'][key] = config['model'][key]

    if args.action_distribution is not None:
        if args.action_distribution == 'truncated_normal':
            from model.custom_action_dist import TruncatedNormal
            ModelCatalog.register_custom_action_dist("truncated_normal", TruncatedNormal)
            config['model']['custom_action_dist'] = 'truncated_normal'
        if args.action_distribution == 'truncated_normal_zero_kl':
            from model.custom_action_dist import TruncatedNormalZeroKL
            ModelCatalog.register_custom_action_dist("truncated_normal_zero_kl", TruncatedNormalZeroKL)
            config['model']['custom_action_dist'] = 'truncated_normal_zero_kl'
        if args.action_distribution == 'beta_alpha_beta':
            from model.custom_action_dist import BetaAlphaBeta
            ModelCatalog.register_custom_action_dist("beta_alpha_beta", BetaAlphaBeta)
            config['model']['custom_action_dist'] = 'beta_alpha_beta'

    if args.action_preprocessing_function == "tanh":
        config['normalize_actions'] = False
        config['clip_actions'] = False

    if args.checkpoint is not None:
        if not os.path.isdir(args.checkpoint) and not os.path.isfile(args.checkpoint):
            checkpoint_path = os.path.join(current_dir, 'trained_networks', args.checkpoint)
        else:
            checkpoint_path = args.checkpoint

        if os.path.isdir(checkpoint_path):
            if os.path.basename(checkpoint_path) == 'checkpoint':
                checkpoint_path = os.path.join(checkpoint_path, 'checkpoint')
            else:
                checkpoint_path = os.path.join(checkpoint_path, 'checkpoint', 'checkpoint')

        if not os.path.isfile(checkpoint_path):
            raise ValueError('Could not find checkpoint {}'.format(checkpoint_path))

        params_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        params_path = os.path.join(params_dir, 'params.json')

        with open(params_path) as params_file:
            checkpoint_config = json.load(params_file)
        config['env_config'] = checkpoint_config['env_config']
        config['train_batch_size'] = checkpoint_config['train_batch_size']
        config['sgd_minibatch_size'] = checkpoint_config['sgd_minibatch_size']

    else:
        checkpoint_path = None
        config['env_config'] = _make_env_config()
        config['train_batch_size'] = int(config['train_batch_size'] * args.batch_size_factor)
        config['sgd_minibatch_size'] = int(config['sgd_minibatch_size'] * args.batch_size_factor)

    if args.logdir is None:
        experiment_path = config['env_config']['experiment_name']
    else:
        experiment_path = os.path.join(args.logdir, config['env_config']['experiment_name'])

    if args.seed is not None:
        config['seed'] = args.seed
        config['env_config']['seed'] = args.seed
        np.random.seed(args.seed)

    if args.num_workers is None:
        config['num_workers'] = int(multiprocessing.cpu_count() * 0.75)
    else:
        config['num_workers'] = args.num_workers

    config['rollout_fragment_length'] = int(config['train_batch_size'] / max(config['num_workers'], 1))

    # define number of threads per worker for parallel execution based on OpenMP
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads_per_worker)

    if config['env_config']['use_splines']:
        if not os.path.isdir(config['env_config']['spline_dir']):
            config['env_config']['spline_dir'] = os.path.join(current_dir, "dataset",
                                                              config['env_config']['spline_dir'])
        if not os.path.isdir(config['env_config']['spline_dir']):
            raise FileNotFoundError("Could not find spline_dir {}".format(config['env_config']['spline_dir']))
        from tracking.envs.tracking_env import TrackingEnvSpline as Env
    else:
        from tracking.envs.tracking_env import TrackingEnv as Env

    env_name = Env.__name__
    config.update(env=env_name)
    tune.register_env(env_name, lambda config_args: Env(**config_args))

    ray.init(dashboard_host='0.0.0.0', include_dashboard=args.use_dashboard, ignore_reinit_error=True,
             logging_level=args.logging_level)
    config['callbacks'] = CustomTrainCallbacks

    if args.num_gpus is not None:
        config['num_gpus'] = args.num_gpus

    stop = {'time_total_s': args.time * 3600}

    experiment = {
        experiment_path: {
            'run': algorithm,
            'env': env_name,
            'stop': stop,
            'config': config,
            'checkpoint_freq': args.iterations_per_checkpoint,
            'checkpoint_at_end': True,
            'keep_checkpoints_num': 10,
            'max_failures': 0,
            'restore': checkpoint_path
        }
    }

    tune.run_experiments(experiment, callbacks=[TBXLoggerCallback()])
