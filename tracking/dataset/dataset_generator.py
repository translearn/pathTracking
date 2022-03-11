#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import datetime
import errno
import glob
import inspect
import json
import logging
import os
import sys
import time
import pybullet as p

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))))))

from tracking.utils.control_rate import ControlRate
from tracking.utils.spline_utils import Spline

RENDERER = {'opengl': 0,
            'egl': 1,
            'cpu': 2,
            'imagegrab': 3}


class DatasetGenerator:
    def __init__(self,
                 input_dir=None,
                 input_dir_balance=None,
                 spline_dir=None,
                 output_dir=None,
                 trajectory_key='setpoints',
                 trajectory_key_visualization=None,
                 use_curvature_for_spline_resampling=False,
                 plot_trajectory=False,
                 plot_spline=False,
                 store_spline=False,
                 visualize_spline=False,
                 visualize_spline_no_reset=False,
                 spline_use_reflection_vectors=False,
                 visualize_trajectory=False,
                 simulate_spline=False,
                 random_agent=False,
                 spline_u_arc_start_range=(0, 0),
                 trajectory_slowdown_factor=None,
                 seed=None,
                 resampling_distance=None,
                 curvature_sampling_distance=None,
                 length_correction_step_size=None,
                 use_normalized_length_correction_step_size=False,
                 train_fraction=None,
                 render=None,
                 renderer=0,
                 render_no_shadows=False,
                 camera_angle=0,
                 use_joint=None,
                 ):
        self._input_dir = input_dir
        self._input_dir_balance = input_dir_balance
        self._spline_dir = spline_dir
        if output_dir is None:
            time_stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            if self._input_dir_balance is None:
                self._output_dir = os.path.join(self._input_dir, "dataset", time_stamp)
            else:
                self._output_dir = os.path.join(self._input_dir_balance, "dataset", time_stamp)
        else:
            self._output_dir = output_dir

        self._trajectory_key = "trajectory_" + trajectory_key
        if trajectory_key_visualization is None:
            self._trajectory_key_visualization = self._trajectory_key
        else:
            self._trajectory_key_visualization = "trajectory_" + trajectory_key_visualization
        self._use_curvature_for_spline_resampling = use_curvature_for_spline_resampling
        self._plot_trajectory = plot_trajectory
        self._plot_spline = plot_spline
        self._store_spline = store_spline
        self._visualize_spline_no_reset = visualize_spline_no_reset
        self._spline_use_reflection_vectors = spline_use_reflection_vectors
        self._spline_reflection_vectors = None
        self._visualize_spline = visualize_spline if not self._visualize_spline_no_reset else True
        self._visualize_trajectory = visualize_trajectory
        self._simulate_spline = simulate_spline
        self._random_agent = random_agent
        self._trajectory_slowdown_factor = trajectory_slowdown_factor
        self._resampling_distance = resampling_distance if resampling_distance is not None else 0.1
        self._curvature_sampling_distance = curvature_sampling_distance \
            if curvature_sampling_distance is not None else 0.3
        self._length_correction_step_size = length_correction_step_size \
            if length_correction_step_size is not None else 0.00002
        self._use_normalized_length_correction_step_size = use_normalized_length_correction_step_size
        self._train_fraction = train_fraction if train_fraction is not None else 0.8
        self._debug_line_buffer = None
        self._cartesian_debug_line_buffer = None

        if self._store_spline:
            self._make_output_dir()

        self._env_config = self._read_env_config()

        if self._visualize_trajectory or self._visualize_spline or self._simulate_spline:
            self._use_gui = True
        else:
            self._use_gui = False

        if self._visualize_trajectory or self._visualize_spline:
            self._switch_gui = True
            if self._simulate_spline:
                raise ValueError("simulate_spline is incompatible with visualize_spline and visualize_trajectory")
        else:
            self._switch_gui = False

        self._render = render

        self._env_config['use_gui'] = self._use_gui
        self._env_config['switch_gui'] = self._switch_gui
        self._env_config['use_target_points'] = False
        self._env_config['store_trajectory'] = False
        self._env_config['use_splines'] = True
        self._env_config['random_agent'] = self._random_agent
        self._env_config['spline_u_arc_start_range'] = spline_u_arc_start_range
        self._env_config['use_control_rate_sleep'] = False
        self._env_config['seed'] = seed

        if self._render:
            self._env_config['render_video'] = True
            self._env_config['renderer'] = renderer
            self._env_config['render_no_shadows'] = render_no_shadows
            self._env_config['camera_angle'] = camera_angle
            self._env_config['video_dir'] = os.path.join(self._output_dir, "video")
            self._env_config['fixed_video_filename'] = True

        if self._plot_trajectory:
            self._env_config['plot_trajectory'] = True
        else:
            self._env_config['plot_trajectory'] = False

        os.environ['OMP_NUM_THREADS'] = '1'
        from tracking.envs.tracking_env import TrackingEnvSpline
        from tracking.utils.spline_plotter import SplinePlotter
        from tracking.utils.trajectory_plotter import TrajectoryPlotter

        self._env = TrackingEnvSpline(**self._env_config)
        self._env._robot_scene.obstacle_wrapper.reset_obstacles()

        self._env._trajectory_manager.spline_use_normalized_length_correction_step_size = \
            self._use_normalized_length_correction_step_size

        self._env._trajectory_manager.spline_length_correction_step_size = \
            self._length_correction_step_size

        self._use_joint = np.array([True] * self._env._robot_scene.num_manip_joints) \
            if use_joint is None else (np.array(use_joint) == True)

        if self._visualize_trajectory or self._visualize_spline:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._env._gui_client_id)
        self._jerk_limits_min_max = np.array([-1 * self._env._robot_scene.max_jerk_linear_interpolation,
                                              self._env._robot_scene.max_jerk_linear_interpolation])

        if self._plot_spline:
            self._spline_plotter = SplinePlotter(plot_norm=True, normalize=True,
                                                 pos_limits=self._env.pos_limits_min_max.T[self._use_joint])
        else:
            self._spline_plotter = None

        if self._plot_trajectory:
            self._trajectory_plotter = TrajectoryPlotter(time_step=self._env.trajectory_time_step,
                                                         control_time_step=self._env._simulation_time_step,
                                                         simulation_time_step=
                                                         self._env._simulation_time_step,
                                                         pos_limits=self._env.pos_limits_min_max.T[self._use_joint],
                                                         vel_limits=self._env._vel_limits_min_max.T[self._use_joint],
                                                         acc_limits=self._env._acc_limits_min_max.T[self._use_joint],
                                                         jerk_limits=self._jerk_limits_min_max.T[self._use_joint],
                                                         torque_limits=None,
                                                         plot_joint=None,
                                                         plot_acc_limits=False,
                                                         plot_actual_values=False,
                                                         plot_computed_actual_values=False,
                                                         plot_actual_torques=False)
        else:
            self._trajectory_plotter = None

        if self._spline_dir is None:
            self._process_trajectory_data()
        else:
            self._process_spline_data()

        self._env.close()

    def _process_trajectory_data(self):
        if self._input_dir_balance is None:
            self._trajectory_data_files = self._get_trajectory_data_files(balance=False)
            self._trajectory_data, self._valid_trajectory_data_files = self._read_json_data(self._trajectory_data_files)
        else:
            self._trajectory_data_files = self._get_trajectory_data_files(balance=True)
            self._trajectory_data, self._valid_trajectory_data_files = \
                self._read_balance_data(self._trajectory_data_files)

        self._num_trajectories = len(self._trajectory_data)

        logging.info("Found {} trajectory files. Ignored {} invalid file(s)".format(self._num_trajectories,
                                                                                    len(self._trajectory_data_files)
                                                                                    - self._num_trajectories))
        spline_config_dict = {'max_dis_between_knots': None,
                              'num_train': None,
                              'num_test': None,
                              'resampling_distance': self._resampling_distance,
                              'curvature_sampling_distance': self._curvature_sampling_distance
                              if self._use_curvature_for_spline_resampling else None,
                              'length_correction_step_size': self._length_correction_step_size * 10,
                              'use_normalized_length_correction_step_size':
                                  self._use_normalized_length_correction_step_size
                              }

        max_distance_between_knots = 0
        num_train = int(self._train_fraction * self._num_trajectories)
        num_test = self._num_trajectories - num_train

        for i in range(self._num_trajectories):
            train_or_test = 'train' if i < num_train else 'test'
            spline_name = os.path.basename(self._valid_trajectory_data_files[i])
            logging.info('Processing trajectory {}/{}: "{}" ({})'.format(i + 1, self._num_trajectories, spline_name,
                                                                         train_or_test))
            curve_data = np.array(self._trajectory_data[i][self._trajectory_key]['positions']).T[self._use_joint, :]

            final_spline, max_distance_spline = self._compute_spline_from_curve_data(curve_data=curve_data)

            max_distance_between_knots = max(max_distance_spline, max_distance_between_knots)

            if self._visualize_trajectory:
                trajectory = np.array(self._trajectory_data[i][self._trajectory_key_visualization]['positions'])
                self._visualize_trajectory_gui(trajectory=trajectory)

            if self._plot_trajectory:
                self._add_joint_trajectory_to_trajectory_plotter(trajectory_dict=
                                                                 self._trajectory_data[i][self._trajectory_key])
                self._trajectory_plotter.display_plot(blocking=not self._plot_spline)

            if self._store_spline:
                final_spline.save_to_json(path=os.path.join(self._output_dir, train_or_test, os.path.basename(
                    self._valid_trajectory_data_files[i])))

            if self._simulate_spline:
                self._run_episode(spline_name=spline_name, spline_data=final_spline,
                                  actions=self._trajectory_data[i]['actions'],
                                  spline_max_distance_between_knots=max_distance_spline)

        spline_config_dict['max_dis_between_knots'] = max_distance_between_knots
        spline_config_dict['num_train'] = num_train
        spline_config_dict['num_test'] = num_test

        if self._store_spline:
            with open(os.path.join(self._output_dir, "spline_config.json"), 'w') as f:
                f.write(json.dumps(spline_config_dict))
                f.flush()

    def _compute_spline_from_curve_data(self, curve_data):
        start = time.time()

        spline = Spline(curve_data=curve_data, curve_data_slicing_step=1,
                        length_correction_step_size=self._length_correction_step_size,
                        use_normalized_length_correction_step_size=
                        self._use_normalized_length_correction_step_size,
                        method="auto",
                        curvature_at_ends=0)

        spline_resampled = spline.copy_with_resampling(resampling_distance=self._resampling_distance,
                                                       use_normalized_resampling_distance=False,
                                                       use_curvature_for_resampling=False,
                                                       length_correction_step_size=
                                                       self._length_correction_step_size,
                                                       use_normalized_length_correction_step_size=
                                                       self._use_normalized_length_correction_step_size,
                                                       )

        if self._use_curvature_for_spline_resampling:
            spline_resampled_curvature = \
                spline_resampled.copy_with_resampling(resampling_distance=self._curvature_sampling_distance,
                                                      use_normalized_resampling_distance=False,
                                                      use_curvature_for_resampling=True,
                                                      length_correction_step_size=self._length_correction_step_size,
                                                      use_normalized_length_correction_step_size=
                                                      self._use_normalized_length_correction_step_size)
            final_spline = spline_resampled_curvature
        else:
            final_spline = spline_resampled

        end = time.time()
        print("Spline Calculation time", end - start)
        max_distance_between_knots = final_spline.max_dis_between_knots
        logging.info("Max distance between knots: {}".format(max_distance_between_knots))

        if self._visualize_spline:
            self._debug_line_buffer = final_spline.visualize(env=self._env, sample_distance=0.1,
                                                             use_normalized_sample_distance=False,
                                                             debug_line_buffer=self._debug_line_buffer)

        if self._plot_spline:
            self._spline_plotter.reset_plotter()
            if self._use_curvature_for_spline_resampling:
                self._spline_plotter.add_joint_spline(spline=spline_resampled_curvature, linestyle=':',
                                                      marker='o')
            else:
                self._spline_plotter.add_joint_spline(spline=spline_resampled, linestyle='-.',
                                                      label='_nolegend_', marker='^')

            self._spline_plotter.display_plot()

        return final_spline, max_distance_between_knots

    def _process_spline_data(self):
        self._spline_data_files = self._get_spline_data_files()

        self._spline_data, self._valid_spline_data_files = self._read_json_data(self._spline_data_files)
        self._num_splines = len(self._spline_data)

        if self._spline_use_reflection_vectors:
            self._load_spline_config()
            reflection_vectors = self._spline_reflection_vectors
        else:
            reflection_vectors = None

        logging.info("Found {} trajectory files. Ignored {} invalid file(s)".format(self._num_splines,
                                                                                    len(self._spline_data_files)
                                                                                    - self._num_splines))

        if self._render and self._visualize_spline_no_reset:
            self._env._reset_video_recorder()

        for i in range(self._num_splines):
            logging.info('Processing reference_spline {}/{}: "{}"'.format(i + 1, self._num_splines,
                                                                          os.path.basename(
                                                                              self._valid_spline_data_files[i])))
            spline = Spline.load_from_dict(self._spline_data[i], reflection_vectors=reflection_vectors)
            spline.reset(random_reflection_vector_index=True)

            if self._visualize_spline or self._visualize_spline_no_reset:
                if self._visualize_spline_no_reset:
                    spline.visualize(env=self._env, sample_distance=0.1,
                                     use_normalized_sample_distance=False,
                                     debug_line_buffer=None)
                    if self._render:
                        time.sleep(0.1)
                        self._env._capture_frame_with_video_recorder(frames=1)
                else:
                    self._debug_line_buffer = spline.visualize(env=self._env, sample_distance=0.1,
                                                               use_normalized_sample_distance=False,
                                                               debug_line_buffer=self._debug_line_buffer)

            if self._plot_spline:
                self._spline_plotter.reset_plotter()
                self._spline_plotter.add_joint_spline(spline=spline)
                self._spline_plotter.display_plot()

        if self._visualize_spline_no_reset:
            if self._render:
                self._env._close_video_recorder()
            else:
                input("Plotted {} spline. Press any key to exit.".format(self._num_splines))

    def _load_spline_config(self):
        spline_config_path = os.path.join(os.path.dirname(self._spline_dir), "spline_config.json")
        if os.path.exists(spline_config_path):
            with open(spline_config_path) as file:
                spline_config_dict = json.load(file)
            if "reflection_vectors" in spline_config_dict:
                self._spline_reflection_vectors = spline_config_dict["reflection_vectors"]

    def _read_json_data(self, json_data_files):
        json_data = []
        valid_json_data_files = []
        for i in range(len(json_data_files)):
            try:
                with open(json_data_files[i]) as file:
                    json_data.append(json.load(file))
                valid_json_data_files.append(json_data_files[i])
            except json.decoder.JSONDecodeError:
                logging.warning("Could not load {}".format(os.path.basename(json_data_files[i])))
        return json_data, valid_json_data_files

    def _read_env_config(self):
        if self._input_dir_balance is None:
            env_config_path = os.path.join(self._input_dir, "env_config.json")
        else:
            env_config_path = os.path.join(self._input_dir_balance, "env_config.json")

        if os.path.isfile(env_config_path):
            with open(env_config_path) as file:
                env_config = json.load(file)
        else:
            raise FileNotFoundError("Could not find file {}".format(env_config_path))

        return env_config

    def _get_trajectory_data_files(self, balance=False):
        trajectory_data_dir = os.path.join(self._input_dir, "trajectory_data")
        search_string = "episode_*.json"
        trajectory_data_files = sorted(glob.glob(os.path.join(trajectory_data_dir, search_string)))
        return trajectory_data_files

    def _get_spline_data_files(self):
        if os.path.exists(self._spline_dir):
            if os.path.isdir:
                spline_data_files = sorted(glob.glob(os.path.join(self._spline_dir, "episode_*.json")))
            else:
                spline_data_files = [self._spline_dir]  # single file
            return spline_data_files
        else:
            raise ValueError("spline_dir {} does not exist!".format(self._spline_dir))

    def _add_joint_trajectory_to_trajectory_plotter(self, trajectory_dict):
        self._trajectory_plotter.reset_plotter(initial_joint_position=trajectory_dict['positions'][0])
        acceleration_setpoints = np.array(trajectory_dict['accelerations'])[::self._env._control_steps_per_action]
        for i in range(1, len(acceleration_setpoints)):
            self._trajectory_plotter.add_data_point(current_acc=acceleration_setpoints[i])

    def _make_output_dir(self):
        for test_or_train in ['test', 'train']:
            if not os.path.exists(os.path.join(self._output_dir, test_or_train)):
                try:
                    os.makedirs(os.path.join(self._output_dir, test_or_train))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

    def _visualize_trajectory_gui(self, trajectory):
        if self._trajectory_slowdown_factor is not None:
            control_rate = ControlRate(1.0 / (self._env._simulation_time_step * self._trajectory_slowdown_factor),
                                       skip_periods=True, debug_mode=False, busy_wait=True)
        for i in range(len(trajectory)):
            self._env._robot_scene.obstacle_wrapper.set_robot_position_in_obstacle_client(
                target_position=trajectory[i])
            if self._trajectory_slowdown_factor is not None:
                control_rate.sleep()

    def _run_episode(self, spline_name, spline_data, actions, spline_max_distance_between_knots=None):
        step = 0
        if spline_max_distance_between_knots is not None:
            self._env._trajectory_manager.spline_max_distance_between_knots = spline_max_distance_between_knots
        self._env._trajectory_manager.add_reference_spline(spline_name=spline_name, spline_data=spline_data)
        self._env.reset(spline_name=spline_name)
        sum_of_rewards = 0
        start_episode_timer = time.time()
        done = False
        if self._trajectory_slowdown_factor is not None:
            control_rate = ControlRate(1.0 / (self._env._trajectory_time_step * self._trajectory_slowdown_factor),
                                       skip_periods=False, debug_mode=False, busy_wait=False)

        while not done:
            if self._random_agent:
                action = None
            else:
                if step < len(actions):
                    action = actions[step]
                else:
                    done = True
            if not done:
                obs, reward, done, info = self._env.step(action=action)
                step += 1
                sum_of_rewards += reward
                if self._trajectory_slowdown_factor is not None:
                    control_rate.sleep()

        end_episode_timer = time.time()
        episode_computation_time = end_episode_timer - start_episode_timer
        logging.info("Episode took %s seconds. Trajectory duration: %s seconds.",
                     episode_computation_time,
                     step * self._env._trajectory_time_step)
        logging.info("Reward: %s", sum_of_rewards)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--use_gui', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--input_dir_balance', type=str, default=None)
    parser.add_argument('--spline_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument("--trajectory_key", default='setpoints', choices=['setpoints', 'measured_actual_values',
                                                                          'computed_actual_values'])
    parser.add_argument("--trajectory_key_visualization",
                        default=None, choices=['setpoints', 'measured_actual_values', 'computed_actual_values'])
    parser.add_argument("--logging_level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--plot_trajectory', action='store_true', default=False)
    parser.add_argument('--random_agent', action='store_true', default=False)
    parser.add_argument('--plot_spline', action='store_true', default=False)
    parser.add_argument('--use_curvature_for_spline_resampling', action='store_true', default=False)
    parser.add_argument('--store_spline', action='store_true', default=False)
    parser.add_argument('--visualize_spline', action='store_true', default=False)
    parser.add_argument('--visualize_spline_no_reset', action='store_true', default=False)
    parser.add_argument('--spline_use_reflection_vectors', action='store_true', default=False)
    parser.add_argument('--visualize_trajectory', action='store_true', default=False)
    parser.add_argument('--simulate_spline', action='store_true', default=False)
    parser.add_argument('--spline_u_arc_start_range', type=json.loads, default=(0, 0))
    parser.add_argument('--trajectory_slowdown_factor', type=float, default=None)
    parser.add_argument('--resampling_distance', type=float, default=None)
    parser.add_argument('--curvature_sampling_distance', type=float, default=None)
    parser.add_argument('--length_correction_step_size', type=float, default=None)
    parser.add_argument('--use_normalized_length_correction_step_size', action='store_true', default=False)
    parser.add_argument('--train_fraction', type=float, default=None)
    parser.add_argument('--render', action='store_true', default=False,
                        help="If set, videos of the generated episodes are recorded.")
    parser.add_argument("--renderer", default='opengl', choices=['opengl', 'egl', 'cpu', 'imagegrab'])
    parser.add_argument('--render_no_shadows', action='store_true', default=False)
    parser.add_argument('--camera_angle', type=int, default=0)
    parser.add_argument('--use_joint', type=json.loads, default=None)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(args.logging_level)

    dataset_generator = DatasetGenerator(input_dir=args.input_dir,
                                         input_dir_balance=args.input_dir_balance,
                                         spline_dir=args.spline_dir,
                                         output_dir=args.output_dir,
                                         trajectory_key=args.trajectory_key,
                                         trajectory_key_visualization=args.trajectory_key_visualization,
                                         plot_trajectory=args.plot_trajectory,
                                         plot_spline=args.plot_spline,
                                         use_curvature_for_spline_resampling=args.use_curvature_for_spline_resampling,
                                         store_spline=args.store_spline,
                                         visualize_spline=args.visualize_spline,
                                         visualize_spline_no_reset=args.visualize_spline_no_reset,
                                         spline_use_reflection_vectors=args.spline_use_reflection_vectors,
                                         visualize_trajectory=args.visualize_trajectory,
                                         simulate_spline=args.simulate_spline,
                                         random_agent=args.random_agent,
                                         spline_u_arc_start_range=args.spline_u_arc_start_range,
                                         trajectory_slowdown_factor=args.trajectory_slowdown_factor,
                                         resampling_distance=args.resampling_distance,
                                         curvature_sampling_distance=args.curvature_sampling_distance,
                                         length_correction_step_size=args.length_correction_step_size,
                                         use_normalized_length_correction_step_size=
                                         args.use_normalized_length_correction_step_size,
                                         train_fraction=args.train_fraction,
                                         render=args.render,
                                         renderer=RENDERER[args.renderer],
                                         render_no_shadows=args.render_no_shadows,
                                         camera_angle=args.camera_angle,
                                         use_joint=args.use_joint,
                                         seed=args.seed)
