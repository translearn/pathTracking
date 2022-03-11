# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import datetime
import logging
import os
import time
from abc import abstractmethod
from functools import partial
from multiprocessing import Pool
import json
import errno
from collections import defaultdict
from itertools import chain
from pathlib import Path
from threading import Thread

import gym
import numpy as np
import pybullet as p

from tracking.robot_scene.real_robot_scene import RealRobotScene
from tracking.robot_scene.simulated_robot_scene import SimRobotScene
from tracking.utils.control_rate import ControlRate
from tracking.utils.trajectory_manager import TrajectoryManager

SIM_TIME_STEP = 1. / 240.
CONTROLLER_TIME_STEP = 1. / 200.
EPISODES_PER_SIMULATION_RESET = 12500  # to avoid out of memory error

# Renderer
OPENGL_GUI_RENDERER = 0
OPENGL_EGL_RENDERER = 1
CPU_TINY_RENDERER = 2


class TrackingBase(gym.Env):
    # Termination reason
    TERMINATION_UNSET = -1
    TERMINATION_SUCCESS = 0  # unused
    TERMINATION_JOINT_LIMITS = 1
    TERMINATION_TRAJECTORY_LENGTH = 2
    TERMINATION_SPLINE_LENGTH = 3
    TERMINATION_SPLINE_DEVIATION = 4
    TERMINATION_ROBOT_STOPPED = 5
    TERMINATION_BALANCING = 6
    TERMINATION_BASE_POS_DEVIATION = 7
    TERMINATION_BASE_ORN_DEVIATION = 8
    TERMINATION_BASE_Z_ANGLE_DEVIATION = 9

    def __init__(self,
                 experiment_name,
                 time_stamp=None,
                 evaluation_dir=None,
                 use_real_robot=False,
                 real_robot_debug_mode=False,
                 use_gui=False,
                 switch_gui=False,
                 control_time_step=None,
                 use_control_rate_sleep=True,
                 use_thread_for_movement=False,
                 use_process_for_movement=False,
                 pos_limit_factor=1,
                 vel_limit_factor=1,
                 acc_limit_factor=1,
                 jerk_limit_factor=1,
                 torque_limit_factor=1,
                 acceleration_after_max_vel_limit_factor=0.01,
                 eval_new_condition_counter=1,
                 log_obstacle_data=False,
                 save_obstacle_data=False,
                 store_actions=False,
                 store_trajectory=False,
                 online_trajectory_duration=8.0,
                 online_trajectory_time_step=0.1,
                 position_controller_time_constants=None,
                 plot_computed_actual_values=False,
                 plot_actual_torques=False,
                 robot_scene=0,
                 obstacle_scene=0,
                 activate_obstacle_collisions=False,
                 observed_link_point_scene=0,
                 obstacle_use_computed_actual_values=False,
                 visualize_bounding_spheres=False,
                 check_braking_trajectory_collisions=False,
                 collision_check_time=None,
                 check_braking_trajectory_observed_points=False,
                 check_braking_trajectory_closest_points=True,
                 check_braking_trajectory_torque_limits=False,
                 closest_point_safety_distance=0.1,
                 observed_point_safety_distance=0.1,
                 use_target_points=False,
                 target_point_cartesian_range_scene=0,
                 target_point_relative_pos_scene=0,
                 target_point_radius=0.05,
                 target_point_sequence=0,
                 target_point_reached_reward_bonus=0.0,
                 target_point_use_actual_position=False,
                 obs_add_target_point_pos=False,
                 obs_add_target_point_relative_pos=False,
                 use_splines=False,
                 visualize_action_spline=False,
                 spline_u_arc_start_range=(0, 0),
                 spline_u_arc_diff_min=1.0,
                 spline_u_arc_diff_max=1.0,
                 spline_deviation_weighting_factors=None,
                 spline_speed_range=None,
                 spline_random_speed_per_time_step=False,
                 spline_speed=None,
                 spline_normalize_duration=False,
                 spline_use_reflection_vectors=False,
                 spline_dir=None,
                 spline_config_path=None,
                 spline_termination_max_deviation=None,
                 spline_termination_extra_time_steps=None,
                 spline_cartesian_deviation_max_threshold=0.1,
                 spline_use_actual_position=False,
                 spline_compute_total_spline_metrics=False,
                 punish_spline_max_cartesian_deviation=False,
                 punish_spline_mean_cartesian_deviation=False,
                 plot_spline=False,
                 sphere_balancing_mode=False,
                 balancing_sphere_dev_min_max=None,
                 terminate_on_balancing_sphere_deviation=False,
                 terminate_balancing_sphere_not_on_board=False,
                 floating_robot_base=False,
                 robot_base_balancing_mode=False,
                 balancing_robot_base_max_pos_deviation=0.2,  # in meter
                 balancing_robot_base_max_orn_deviation=1.0,  # normalized 0: min, 1: max
                 balancing_robot_base_max_z_angle_deviation_rad=0.52,  # 0 to pi; 0.52 rad -> 30 deg
                 balancing_robot_base_punish_last_cartesian_action_point=False,
                 terminate_on_balancing_robot_base_pos_deviation=False,
                 terminate_on_balancing_robot_base_orn_deviation=False,
                 terminate_on_balancing_robot_base_z_angle_deviation=False,
                 target_link_name=None,
                 target_link_offset=None,
                 no_self_collision=False,
                 terminate_on_robot_stop=False,
                 use_controller_target_velocities=False,
                 time_step_fraction_sleep_observation=0.0,
                 seed=None,
                 solver_iterations=None,
                 logging_level="WARNING",
                 random_agent=False,
                 **kwargs):

        self._fixed_seed = None
        self.set_seed(seed)
        if evaluation_dir is None:
            evaluation_dir = os.path.join(Path.home(), "tracking_evaluation")
        self._time_stamp = time_stamp
        logging.getLogger().setLevel(logging_level)
        if self._time_stamp is None:
            self._time_stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

        self._experiment_name = experiment_name
        self._evaluation_dir = os.path.join(evaluation_dir, self.__class__.__name__,
                                            self._experiment_name, self._time_stamp)
        self._pid = os.getpid()

        if solver_iterations is None:
            self._solver_iterations = 150
        else:
            self._solver_iterations = solver_iterations

        self._target_link_name = target_link_name
        self._use_real_robot = use_real_robot
        self._use_gui = use_gui
        self._switch_gui = switch_gui
        self._use_control_rate_sleep = use_control_rate_sleep
        self._num_physic_clients = 0
        self._gui_client_id = None

        self._init_physic_clients()

        if control_time_step is None:
            self._control_time_step = CONTROLLER_TIME_STEP if self._use_real_robot else SIM_TIME_STEP
        else:
            self._control_time_step = control_time_step

        self._simulation_time_step = SIM_TIME_STEP
        self._control_step_counter = 0
        self._episode_counter = 0

        self._obstacle_scene = obstacle_scene
        self._activate_obstacle_collisions = activate_obstacle_collisions
        self._observed_link_point_scene = observed_link_point_scene
        self._visualize_bounding_spheres = visualize_bounding_spheres
        self._log_obstacle_data = log_obstacle_data
        self._save_obstacle_data = save_obstacle_data
        self._robot_scene_config = robot_scene
        self._check_braking_trajectory_collisions = check_braking_trajectory_collisions
        self._collision_check_time = collision_check_time
        self._check_braking_trajectory_observed_points = check_braking_trajectory_observed_points
        self._check_braking_trajectory_closest_points = check_braking_trajectory_closest_points
        self._check_braking_trajectory_torque_limits = check_braking_trajectory_torque_limits
        self._closest_point_safety_distance = closest_point_safety_distance
        self._observed_point_safety_distance = observed_point_safety_distance
        self._use_target_points = use_target_points
        self._target_point_cartesian_range_scene = target_point_cartesian_range_scene
        self._target_point_relative_pos_scene = target_point_relative_pos_scene
        self._target_point_radius = target_point_radius
        self._target_point_sequence = target_point_sequence
        self._target_point_reached_reward_bonus = target_point_reached_reward_bonus
        self._target_point_use_actual_position = target_point_use_actual_position
        self._obs_add_target_point_pos = obs_add_target_point_pos
        self._obs_add_target_point_relative_pos = obs_add_target_point_relative_pos
        self._use_splines = use_splines
        self._visualize_action_spline = visualize_action_spline
        self._spline_u_arc_start_range = spline_u_arc_start_range
        self._spline_u_arc_diff_min = spline_u_arc_diff_min
        self._spline_u_arc_diff_max = spline_u_arc_diff_max
        self._spline_deviation_weighting_factors = spline_deviation_weighting_factors
        self._spline_speed_range = spline_speed_range
        self._spline_speed_fixed = False
        if spline_speed is not None:
            self._spline_speed = spline_speed
            self._spline_speed_fixed = True
            logging.info("Using a fixed spline speed: %s", self._spline_speed)
        else:
            self._spline_speed = None
        self._spline_random_speed_per_time_step = spline_random_speed_per_time_step
        self._spline_normalize_duration = spline_normalize_duration
        self._spline_use_reflection_vectors = spline_use_reflection_vectors
        self._spline_dir = spline_dir
        self._spline_config_path = spline_config_path
        self._spline_termination_max_deviation = spline_termination_max_deviation
        self._spline_termination_extra_time_steps = spline_termination_extra_time_steps
        self._spline_cartesian_deviation_max_threshold = spline_cartesian_deviation_max_threshold
        self._spline_use_actual_position = spline_use_actual_position
        self._spline_compute_total_spline_metrics = spline_compute_total_spline_metrics
        self._punish_spline_max_cartesian_deviation = punish_spline_max_cartesian_deviation
        self._punish_spline_mean_cartesian_deviation = punish_spline_mean_cartesian_deviation
        self._plot_spline = plot_spline
        self._sphere_balancing_mode = sphere_balancing_mode
        self._balancing_sphere_dev_min_max = balancing_sphere_dev_min_max
        self._terminate_on_balancing_sphere_deviation = terminate_on_balancing_sphere_deviation
        if self._terminate_on_balancing_sphere_deviation and self._balancing_sphere_dev_min_max is None:
            logging.warning("terminate_on_balancing_sphere_deviation is ignored "
                            "if balancing_sphere_dev_min_max is None")
        self._terminate_balancing_sphere_not_on_board = terminate_balancing_sphere_not_on_board
        self._floating_robot_base = floating_robot_base
        self._robot_base_balancing_mode = robot_base_balancing_mode
        self._balancing_robot_base_max_pos_deviation = balancing_robot_base_max_pos_deviation
        self._balancing_robot_base_max_orn_deviation = balancing_robot_base_max_orn_deviation
        self._balancing_robot_base_max_z_angle_deviation_rad = balancing_robot_base_max_z_angle_deviation_rad
        self._balancing_robot_base_punish_last_cartesian_action_point = \
            balancing_robot_base_punish_last_cartesian_action_point
        self._terminate_on_balancing_robot_base_pos_deviation = terminate_on_balancing_robot_base_pos_deviation
        self._terminate_on_balancing_robot_base_orn_deviation = terminate_on_balancing_robot_base_orn_deviation
        self._terminate_on_balancing_robot_base_z_angle_deviation = terminate_on_balancing_robot_base_z_angle_deviation
        self._no_self_collision = no_self_collision
        self._terminate_on_robot_stop = terminate_on_robot_stop
        self._use_controller_target_velocities = use_controller_target_velocities
        self._trajectory_time_step = online_trajectory_time_step
        self._position_controller_time_constants = position_controller_time_constants
        self._plot_computed_actual_values = plot_computed_actual_values
        self._plot_actual_torques = plot_actual_torques
        self._pos_limit_factor = pos_limit_factor
        self._vel_limit_factor = vel_limit_factor
        self._acc_limit_factor = acc_limit_factor
        self._jerk_limit_factor = jerk_limit_factor
        self._torque_limit_factor = torque_limit_factor
        self._acceleration_after_max_vel_limit_factor = acceleration_after_max_vel_limit_factor
        self._online_trajectory_duration = online_trajectory_duration
        self._eval_new_condition_counter = eval_new_condition_counter
        self._store_actions = store_actions
        self._store_trajectory = store_trajectory
        self._target_link_offset = target_link_offset
        self._real_robot_debug_mode = real_robot_debug_mode
        self._random_agent = random_agent

        self._network_prediction_part_done = None
        self._use_thread_for_movement = use_thread_for_movement
        self._use_process_for_movement = use_process_for_movement
        if self._use_thread_for_movement and self._use_process_for_movement:
            raise ValueError("use_thread_for_movement and use_process_for_movement are not "
                             "allowed to be True simultaneously")
        self._use_movement_thread_or_process = self._use_thread_for_movement or self._use_process_for_movement
        if self._use_movement_thread_or_process and not self._use_control_rate_sleep:
            logging.warning("use_movement_thread_or_process without use_control_rate_sleep == True")
        if self._use_real_robot and not self._use_movement_thread_or_process:
            raise ValueError("use_real_robot requires either use_thread_for_movement or use_process_for_movement")
        if self._real_robot_debug_mode and not self._use_real_robot:
            raise ValueError("real_robot_debug_mode requires use_real_robot")

        self._time_step_fraction_sleep_observation = time_step_fraction_sleep_observation
        # 0..1; fraction of the time step,  the main thread sleeps before getting the next observation;
        # only relevant if self._use_real_robot == True
        if time_step_fraction_sleep_observation != 0:
            logging.info("time_step_fraction_sleep_observation %s", self._time_step_fraction_sleep_observation)
        self._obstacle_use_computed_actual_values = obstacle_use_computed_actual_values
        # use computed actual values to determine the distance between the robot and obstacles and as initial point
        # for torque simulations -> advantage: can be computed in advance, no measurements -> real-time capable
        # disadvantage: controller model might introduce inaccuracies
        if self._use_movement_thread_or_process and not self._obstacle_use_computed_actual_values:
            raise ValueError("Real-time execution requires obstacle_use_computed_actual_values to be True")

        if self._use_movement_thread_or_process:
            if self._use_thread_for_movement:
                logging.info("Using movement thread")
            else:
                logging.info("Using movement process")

        if self._use_process_for_movement:
            self._movement_process_pool = Pool(processes=1)
        else:
            self._movement_process_pool = None

        self._model_actual_values = self._use_movement_thread_or_process or self._obstacle_use_computed_actual_values \
                                    or self._plot_computed_actual_values or (self._use_real_robot and self._use_gui)

        if not self._use_movement_thread_or_process and self._control_time_step != self._simulation_time_step:
            raise ValueError("If no movement thread or process is used, the control time step must equal the control "
                             "time step of the obstacle client")

        self._start_position = None
        self._start_velocity = None
        self._start_acceleration = None
        self._position_deviation = None
        self._acceleration_deviation = None
        self._current_trajectory_point_index = None
        self._trajectory_successful = None
        self._total_reward = None
        self._episode_length = None
        self._action_list = []
        self._last_action = None
        self._termination_reason = self.TERMINATION_UNSET
        self._movement_thread = None
        self._movement_process = None
        self._brake = False

        self._adaptation_punishment = None
        self._end_min_distance = None
        self._end_max_torque = None  # for (optional) reward calculations
        self._punish_end_max_torque = False  # set in rewards.py
        self._spline_debug_line_buffer = None

        self._init_simulation()

        if self._gui_client_id is not None:
            # deactivate rendering temporarily to reduce the computational effort for the additional process that
            # ray spawns to detect the observation space and the action space
            # rendering is activated the first time that reset is called
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._gui_client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._gui_client_id)

    def _init_physic_clients(self):
        self._num_physic_clients = 0

        if self._render_video:
            pybullet_options = "--width={} --height={}".format(self._video_width, self._video_height)
        else:
            pybullet_options = ""

        if self._use_gui and not self._switch_gui:
            self._simulation_client_id = p.connect(p.GUI, options=pybullet_options)
            self._gui_client_id = self._simulation_client_id
            self._num_physic_clients += 1
        else:
            if not self._use_real_robot:
                self._simulation_client_id = p.connect(p.DIRECT, options=pybullet_options)
                self._num_physic_clients += 1
            else:
                self._simulation_client_id = None

        self._egl_plugin = None

        if self._simulation_client_id is not None:
            if self._renderer == OPENGL_GUI_RENDERER and self._render_video and not self._use_gui:
                raise ValueError("renderer==OPENGL_GUI_RENDERER requires use_gui==True")
            if self._renderer == OPENGL_GUI_RENDERER or self._renderer == OPENGL_EGL_RENDERER:
                self._pybullet_renderer = p.ER_BULLET_HARDWARE_OPENGL
                if self._renderer == OPENGL_EGL_RENDERER and self._render_video:
                    import pkgutil
                    egl_renderer = pkgutil.get_loader('eglRenderer')
                    logging.warning(
                        "The usage of the egl renderer might lead to a segmentation fault on systems without "
                        "a GPU.")
                    if egl_renderer:
                        self._egl_plugin = p.loadPlugin(egl_renderer.get_filename(), "_eglRendererPlugin")
                    else:
                        self._egl_plugin = p.loadPlugin("eglRendererPlugin")
            else:
                self._pybullet_renderer = p.ER_TINY_RENDERER
        else:
            self._pybullet_renderer = None

        if self._use_gui and self._switch_gui:
            self._obstacle_client_id = p.connect(p.GUI, options=pybullet_options)
            self._gui_client_id = self._obstacle_client_id
        else:
            self._obstacle_client_id = p.connect(p.DIRECT)
        self._num_physic_clients += 1

    def _init_simulation(self):
        # reset the physics engine
        for i in range(self._num_physic_clients):
            p.resetSimulation(physicsClientId=i)  # to free memory

        if self._render_video and self._use_gui and self._switch_gui:
            capture_frame_function = partial(self._capture_frame_with_video_recorder, frames=2)
        else:
            capture_frame_function = None

        # robot scene settings
        robot_scene_parameters = {'simulation_client_id': self._simulation_client_id,
                                  'simulation_time_step': self._simulation_time_step,
                                  'obstacle_client_id': self._obstacle_client_id,
                                  'trajectory_time_step': self._trajectory_time_step,
                                  'use_real_robot': self._use_real_robot,
                                  'robot_scene': self._robot_scene_config,
                                  'obstacle_scene': self._obstacle_scene,
                                  'visual_mode': self._use_gui or self._render_video,
                                  'capture_frame_function': capture_frame_function,
                                  'activate_obstacle_collisions': self._activate_obstacle_collisions,
                                  'observed_link_point_scene': self._observed_link_point_scene,
                                  'log_obstacle_data': self._log_obstacle_data,
                                  'visualize_bounding_spheres': self._visualize_bounding_spheres,
                                  'acc_range_function': self.compute_next_acc_min_and_next_acc_max,
                                  'acc_braking_function': self.acc_braking_function,
                                  'check_braking_trajectory_collisions': self._check_braking_trajectory_collisions,
                                  'collision_check_time': self._collision_check_time,
                                  'check_braking_trajectory_observed_points':
                                      self._check_braking_trajectory_observed_points,
                                  'check_braking_trajectory_closest_points':
                                      self._check_braking_trajectory_closest_points,
                                  'check_braking_trajectory_torque_limits':
                                      self._check_braking_trajectory_torque_limits,
                                  'closest_point_safety_distance': self._closest_point_safety_distance,
                                  'observed_point_safety_distance': self._observed_point_safety_distance,
                                  'use_target_points': self._use_target_points,
                                  'target_point_cartesian_range_scene': self._target_point_cartesian_range_scene,
                                  'target_point_relative_pos_scene': self._target_point_relative_pos_scene,
                                  'target_point_radius': self._target_point_radius,
                                  'target_point_sequence': self._target_point_sequence,
                                  'target_point_reached_reward_bonus': self._target_point_reached_reward_bonus,
                                  'target_point_use_actual_position': self._target_point_use_actual_position,
                                  'use_splines': self._use_splines,
                                  'no_self_collision': self._no_self_collision,
                                  'floating_robot_base': self._floating_robot_base,
                                  'target_link_name': self._target_link_name,
                                  'target_link_offset': self._target_link_offset,
                                  'pos_limit_factor': self._pos_limit_factor,
                                  'vel_limit_factor': self._vel_limit_factor,
                                  'acc_limit_factor': self._acc_limit_factor,
                                  'jerk_limit_factor': self._jerk_limit_factor,
                                  'torque_limit_factor': self._torque_limit_factor,
                                  'use_controller_target_velocities': self._use_controller_target_velocities,
                                  'reward_maximum_relevant_distance': self.reward_maximum_relevant_distance,
                                  'sphere_balancing_mode': self._sphere_balancing_mode,
                                  'robot_base_balancing_mode': self._robot_base_balancing_mode,
                                  }

        if self._use_real_robot:
            self._robot_scene = RealRobotScene(real_robot_debug_mode=self._real_robot_debug_mode,
                                               **robot_scene_parameters)
        else:
            self._robot_scene = SimRobotScene(**robot_scene_parameters)

        self._num_manip_joints = self._robot_scene.num_manip_joints
        if self._position_controller_time_constants is None:
            if self._use_controller_target_velocities:
                self._position_controller_time_constants = [0.0005] * self._num_manip_joints
            else:
                self._position_controller_time_constants = [0.0372] * self._num_manip_joints

        # trajectory manager settings
        self._trajectory_manager = TrajectoryManager(trajectory_time_step=self._trajectory_time_step,
                                                     trajectory_duration=self._online_trajectory_duration,
                                                     obstacle_wrapper=self._robot_scene.obstacle_wrapper,
                                                     use_splines=self._use_splines,
                                                     spline_u_arc_start_range=self._spline_u_arc_start_range,
                                                     spline_u_arc_diff_min=self._spline_u_arc_diff_min,
                                                     spline_u_arc_diff_max=self._spline_u_arc_diff_max,
                                                     spline_deviation_weighting_factors=
                                                     self._spline_deviation_weighting_factors,
                                                     spline_normalize_duration=self._spline_normalize_duration,
                                                     spline_dir=self._spline_dir,
                                                     spline_config_path=self._spline_config_path,
                                                     visualize_action_spline=self._visualize_action_spline,
                                                     spline_termination_max_deviation=
                                                     self._spline_termination_max_deviation,
                                                     spline_termination_extra_time_steps=
                                                     self._spline_termination_extra_time_steps,
                                                     spline_compute_cartesian_deviation=
                                                     self._punish_spline_mean_cartesian_deviation
                                                     or self._punish_spline_max_cartesian_deviation
                                                     or self._obs_add_target_point_pos
                                                     or self._obs_add_target_point_relative_pos
                                                     or self._balancing_robot_base_punish_last_cartesian_action_point,
                                                     spline_use_reflection_vectors=self._spline_use_reflection_vectors,
                                                     env=self)

        self._robot_scene.compute_actual_joint_limits()
        self._control_steps_per_action = int(round(self._trajectory_time_step / self._control_time_step))
        self._obstacle_client_update_steps_per_action = int(round(self._trajectory_time_step /
                                                                  self._simulation_time_step))

        logging.info("Trajectory time step: " + str(self._trajectory_time_step))

        # calculate model coefficients to estimate actual values if required
        if self._model_actual_values:
            self._trajectory_manager.compute_controller_model_coefficients(self._position_controller_time_constants,
                                                                           self._simulation_time_step)

        self._zero_joint_vector_list = [0.0] * self._num_manip_joints
        self._zero_joint_vector_array = np.array(self._zero_joint_vector_list)

        if (self._use_movement_thread_or_process or self._use_gui) and self._use_control_rate_sleep:
            try:
                log_level = logging.root.level
                import rospy
                rospy.init_node("tracking_control_rate", anonymous=True, disable_signals=True)
                from importlib import reload  
                reload(logging)
                logging.basicConfig()
                logging.getLogger().setLevel(log_level)
            except: 
                pass
            self._control_rate = ControlRate(1. / self._control_time_step, skip_periods=True, debug_mode=False)
        else:
            self._control_rate = None

        for i in range(self._num_physic_clients):
            p.setGravity(0, 0, -9.81, physicsClientId=i)
            p.setPhysicsEngineParameter(numSolverIterations=self._solver_iterations, physicsClientId=i)
            p.setTimeStep(self._simulation_time_step, physicsClientId=i)

    def reset(self, spline_name=None):
        self._episode_counter += 1
        if self._episode_counter == 1 and self._gui_client_id is not None:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._gui_client_id)
            if self._render_video and not self._renderer == self.IMAGEGRAB_RENDERER:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self._gui_client_id)

        if self._episode_counter % EPISODES_PER_SIMULATION_RESET == 0:
            self._disconnect_physic_clients()
            self._init_physic_clients()
            self._init_simulation()

        self._control_step_counter = 0

        self._total_reward = 0
        self._episode_length = 0
        self._trajectory_successful = True
        self._current_trajectory_point_index = 0
        self._action_list = []

        self._network_prediction_part_done = False

        if self._floating_robot_base and self._simulation_client_id is not None:
            p.resetBasePositionAndOrientation(self._robot_scene.robot_id, posObj=[0, 0, 0],
                                              ornObj=[0, 0, 0, 1], physicsClientId=self._simulation_client_id)

        get_new_setup = (((self._episode_counter-1) % self._eval_new_condition_counter) == 0)

        self._robot_scene.obstacle_wrapper.reset_obstacles()

        duration_multiplier = None
        if self._spline_speed_range is not None:
            if self._spline_random_speed_per_time_step and not self._spline_speed_fixed:
                duration_multiplier = 1 / (0.5 * (self._spline_speed_range[0] + self._spline_speed_range[1]))
            else:
                duration_multiplier = 1 / self._spline_speed

        self._trajectory_manager.reset(get_new_trajectory=get_new_setup, spline_name=spline_name,
                                       duration_multiplier=duration_multiplier)
        if self._use_splines and self._use_gui and not self._switch_gui:
            self._spline_debug_line_buffer = \
                self._trajectory_manager.reference_spline.visualize(env=self,
                                                                    sample_distance=0.05,
                                                                    use_normalized_sample_distance=False,
                                                                    debug_line_buffer=self._spline_debug_line_buffer,
                                                                    visualize_knots=True,
                                                                    visualize_knots_orn=False,
                                                                    physics_client_id=self._gui_client_id)

        self._start_position = np.array(self._get_trajectory_start_position())
        self._start_velocity = self._zero_joint_vector_array
        self._start_acceleration = self._zero_joint_vector_array

        if self._use_real_robot:
            logging.info("Starting position: %s", self._start_position)
        else:
            logging.debug("Starting position: %s", self._start_position)
        self._robot_scene.pose_manipulator(self._start_position)
        self._robot_scene.obstacle_wrapper.reset(start_position=self._start_position)
        self._robot_scene.obstacle_wrapper.update(target_position=self._start_position,
                                                  target_velocity=self._start_velocity,
                                                  target_acceleration=self._start_acceleration,
                                                  actual_position=self._start_position,
                                                  actual_velocity=self._start_velocity)

        self._reset_plotter(self._start_position)
        self._add_computed_actual_position_to_plot(self._start_position, self._start_velocity,
                                                   self._start_acceleration)

        if not self._use_real_robot:
            self._add_actual_position_to_plot(self._start_position)

        if not self._use_real_robot:
            # the initial torques are not zero due to gravity
            self._robot_scene.set_motor_control(target_positions=self._start_position,
                                                target_velocities=self._start_velocity,
                                                target_accelerations=self._start_acceleration,
                                                physics_client_id=self._simulation_client_id)
            p.stepSimulation(physicsClientId=self._simulation_client_id)
        if self._plot_actual_torques and not self._use_real_robot:
            actual_joint_torques = self._robot_scene.get_actual_joint_torques()
            self._add_actual_torques_to_plot(actual_joint_torques)
        else:
            self._add_actual_torques_to_plot(self._zero_joint_vector_list)

        self._calculate_safe_acc_range(self._start_position, self._start_velocity, self._start_acceleration,
                                       self._current_trajectory_point_index)

        self._termination_reason = self.TERMINATION_UNSET
        self._last_action = None
        self._network_prediction_part_done = False
        self._movement_thread = None
        self._movement_process = None
        self._brake = False
        self._end_min_distance = None
        self._end_max_torque = None
        self._adaptation_punishment = None

        return None

    def step(self, action):
        self._episode_length += 1
        self._robot_scene.clear_last_action()

        if self._random_agent:
            action = np.random.uniform(-1, 1, self.action_space.shape)
            # overwrite the desired action with a random action
        else:
            action = np.asarray(action, dtype=np.float64)

        if self._store_actions or self._store_trajectory:
            self._action_list.append(action)

        logging.debug("Action %s: %s", self._episode_length - 1, action)

        end_acceleration, controller_setpoints, obstacle_client_update_setpoints, preprocessed_action, \
            action_info, robot_stopped = self._compute_controller_setpoints_from_action(action)

        if self._store_trajectory or self._plot_spline or self._spline_compute_total_spline_metrics:
            for i in range(len(controller_setpoints['positions'])):
                self._add_generated_trajectory_control_point(controller_setpoints['positions'][i],
                                                             controller_setpoints['velocities'][i],
                                                             controller_setpoints['accelerations'][i])
        
        for i in range(len(obstacle_client_update_setpoints['positions'])):

            if self._model_actual_values:

                last_position_setpoint = self._start_position if i == 0 else obstacle_client_update_setpoints[
                    'positions'][i - 1]
                computed_position_is = self._trajectory_manager.model_position_controller_to_compute_actual_values(
                    current_setpoint=obstacle_client_update_setpoints['positions'][i],
                    last_setpoint=last_position_setpoint)

                last_velocity_setpoint = self._start_velocity if i == 0 else obstacle_client_update_setpoints[
                    'velocities'][i - 1]
                computed_velocity_is = self._trajectory_manager.model_position_controller_to_compute_actual_values(
                    current_setpoint=obstacle_client_update_setpoints['velocities'][i],
                    last_setpoint=last_velocity_setpoint, key='velocities')

                last_acceleration_setpoint = self._start_acceleration if i == 0 else obstacle_client_update_setpoints[
                    'accelerations'][i - 1]
                computed_acceleration_is = self._trajectory_manager.model_position_controller_to_compute_actual_values(
                    current_setpoint=obstacle_client_update_setpoints['accelerations'][i],
                    last_setpoint=last_acceleration_setpoint, key='accelerations')

                self._add_computed_actual_trajectory_control_point(computed_position_is,
                                                                   computed_velocity_is,
                                                                   computed_acceleration_is)
                self._add_computed_actual_position_to_plot(computed_position_is, computed_velocity_is,
                                                           computed_acceleration_is)

                if self._use_movement_thread_or_process or self._obstacle_use_computed_actual_values:
                    self._robot_scene.obstacle_wrapper.update(
                        target_position=obstacle_client_update_setpoints['positions'][i],
                        target_velocity=obstacle_client_update_setpoints['velocities'][i],
                        target_acceleration=obstacle_client_update_setpoints['accelerations'][i],
                        actual_position=computed_position_is,
                        actual_velocity=computed_velocity_is)

        if self._use_splines and not (self._spline_use_actual_position
                                      and not self._obstacle_use_computed_actual_values):
            if self._spline_use_actual_position and self._obstacle_use_computed_actual_values:
                # use computed actual values to compute the action spline
                knot_data = np.array(self._get_computed_actual_trajectory_control_point(
                    index=-1 * (len(obstacle_client_update_setpoints['positions']) + 1), start_at_index=True,
                    key='positions'))
            else:
                # use position setpoints to compute the action spline
                knot_data = np.concatenate((self._start_position.reshape(1, -1),
                                            obstacle_client_update_setpoints['positions']), axis=0)
            self._trajectory_manager.generate_action_spline(
                knot_data=knot_data[:, self._robot_scene.spline_joint_mask].T)

        if self._control_rate is not None and self._episode_length == 1:
            # start the control phase and compute the precomputation time
            if hasattr(self._control_rate, 'start_control_phase'):
                self._control_rate.start_control_phase()
            else:
                self._control_rate.sleep()
        
        if self._use_movement_thread_or_process:
            if self._use_thread_for_movement:
                movement_thread = Thread(target=self._execute_robot_movement,
                                         kwargs=dict(controller_setpoints=controller_setpoints))
                if self._movement_thread is not None:
                    self._movement_thread.join()
                movement_thread.start()
                self._movement_thread = movement_thread
            if self._use_process_for_movement:
                control_rate = None if self._control_rate is None else self._control_rate.control_rate
                control_function = self._robot_scene.send_command_to_trajectory_controller \
                    if not self._real_robot_debug_mode else None
                fifo_path = self._robot_scene.FIFO_PATH if not self._real_robot_debug_mode else None
                if self._movement_process is not None:
                    last_time = self._movement_process.get()
                else:
                    last_time = None

                self._movement_process = \
                    self._movement_process_pool.apply_async(func=self._execute_robot_movement_as_process,
                                                            kwds=dict(control_function=control_function,
                                                                      fifo_path=fifo_path,
                                                                      controller_position_setpoints=
                                                                      controller_setpoints['positions'],
                                                                      control_rate=control_rate,
                                                                      last_time=last_time))

                time.sleep(0.002)
                # the movement process will start faster if the main process sleeps during the start-up phase

            movement_info = {'average': {}, 'min': {}, 'max': {}}
        else:
            self._movement_thread = None
            movement_info = self._execute_robot_movement(controller_setpoints=controller_setpoints)

        self._start_position = obstacle_client_update_setpoints['positions'][-1]
        self._start_velocity = obstacle_client_update_setpoints['velocities'][-1]
        self._start_acceleration = end_acceleration

        self._add_generated_trajectory_point(self._start_position, self._start_velocity, self._start_acceleration)

        self._current_trajectory_point_index += 1
        self._last_action = preprocessed_action  # store the last action for reward calculation

        self._calculate_safe_acc_range(self._start_position, self._start_velocity, self._start_acceleration,
                                       self._current_trajectory_point_index)

        # sleep for a specified part of the time_step before getting the observation
        if self._time_step_fraction_sleep_observation != 0:
            time.sleep(self._trajectory_time_step * self._time_step_fraction_sleep_observation)

        observation, reward, done, info = self._process_action_outcome(movement_info, action_info, robot_stopped)

        if not self._network_prediction_part_done:
            self._total_reward += reward
        else:
            done = True

        if done:
            self._network_prediction_part_done = True

        if not self._network_prediction_part_done:
            self._prepare_for_next_action()
        else:
            if not self._use_real_robot or robot_stopped:

                if self._movement_thread is not None:
                    self._movement_thread.join()
                if self._movement_process is not None:
                    self._movement_process.get()

                self._robot_scene.prepare_for_end_of_episode()
                self._prepare_for_end_of_episode()
                observation, reward, _, info = self._process_end_of_episode(observation, reward, done, info)

                if self._store_actions:
                    self._store_action_list()
                if self._store_trajectory:
                    self._store_trajectory_data()
            else:
                self._brake = True  # slow down the robot prior to stopping the episode
                done = False

        return observation, reward, done, dict(info)

    def _execute_robot_movement(self, controller_setpoints):
        # executed in real-time if required
        actual_joint_torques_rel_abs_list = []

        use_actual_positions_for_action_spline = self._use_splines and \
            (self._spline_use_actual_position and not self._obstacle_use_computed_actual_values)

        for i in range(len(controller_setpoints['positions'])):

            if self._control_rate is not None:
                self._control_rate.sleep()

            self._robot_scene.set_motor_control(controller_setpoints['positions'][i],
                                                target_velocities=controller_setpoints['velocities'][i],
                                                target_accelerations=controller_setpoints['accelerations'][i],
                                                computed_position_is=controller_setpoints['positions'][i],
                                                computed_velocity_is=controller_setpoints['velocities'][i])

            if not self._use_real_robot:
                self._sim_step()
                actual_joint_torques = self._robot_scene.get_actual_joint_torques()
                actual_joint_torques_rel_abs = np.abs(normalize_joint_values(actual_joint_torques,
                                                                             self._robot_scene.max_torques))
                actual_joint_torques_rel_abs_list.append(actual_joint_torques_rel_abs)

                if self._plot_actual_torques:
                    self._add_actual_torques_to_plot(actual_joint_torques)

            actual_position = None

            if (not self._use_movement_thread_or_process and not self._obstacle_use_computed_actual_values) or \
                    use_actual_positions_for_action_spline:
                actual_position, actual_velocity = self._robot_scene.get_actual_joint_position_and_velocity()

                if self._store_trajectory or use_actual_positions_for_action_spline:
                    actual_acceleration = (actual_velocity - np.array(
                        self._get_measured_actual_trajectory_control_point(-1, key='velocities'))) / \
                                          self._simulation_time_step
                    self._add_measured_actual_trajectory_control_point(actual_position, actual_velocity,
                                                                       actual_acceleration)

            if not self._use_movement_thread_or_process and not self._obstacle_use_computed_actual_values:
                self._robot_scene.obstacle_wrapper.update(target_position=controller_setpoints['positions'][i],
                                                          target_velocity=controller_setpoints['velocities'][i],
                                                          target_acceleration=controller_setpoints['accelerations'][
                                                              i],
                                                          actual_position=actual_position,
                                                          actual_velocity=actual_velocity)

            if not self._use_real_robot:
                self._add_actual_position_to_plot(actual_position)

        if use_actual_positions_for_action_spline:
            # use measured actual values to compute the action spline
            knot_data = np.array(self._get_measured_actual_trajectory_control_point(
                index=-1 * (len(controller_setpoints['positions']) + 1), start_at_index=True,
                key='positions'), dtype=np.float64)

            self._trajectory_manager.generate_action_spline(knot_data=knot_data.T)

        movement_info = {'average': {}, 'min': {}, 'max': {}}

        if not self._use_real_robot:
            # add torque info to movement_info
            torque_violation = 0.0
            actual_joint_torques_rel_abs = np.array(actual_joint_torques_rel_abs_list)
            if self._punish_end_max_torque and self._end_max_torque is None:
                self._end_max_torque = np.max(actual_joint_torques_rel_abs[-1])
            actual_joint_torques_rel_abs_swap = actual_joint_torques_rel_abs.T
            for j in range(self._num_manip_joints):
                movement_info['average']['joint_{}_torque_abs'.format(j)] = np.mean(
                    actual_joint_torques_rel_abs_swap[j])
                actual_joint_torques_rel_abs_max = np.max(actual_joint_torques_rel_abs_swap[j])
                movement_info['max']['joint_{}_torque_abs'.format(j)] = actual_joint_torques_rel_abs_max
                if actual_joint_torques_rel_abs_max > 1.001:
                    torque_violation = 1.0
                    logging.warning("Torque violation: t = %s Joint: %s Rel torque %s",
                                    (self._episode_length - 1) * self._trajectory_time_step, j,
                                    actual_joint_torques_rel_abs_max)

            movement_info['max']['joint_torque_violation'] = torque_violation
            movement_info['average']['joint_torque_violation'] = torque_violation

        return movement_info

    @staticmethod
    def _execute_robot_movement_as_process(control_function, fifo_path, controller_position_setpoints,
                                           control_rate=None, last_time=None):
        if control_rate is not None:
            control_rate = ControlRate(control_rate=control_rate, skip_periods=False, debug_mode=False,
                                       last_time=last_time, busy_wait=True)
        
        fifo_in = None
        if fifo_path is not None:
            fifo_in = os.open(fifo_path, os.O_WRONLY) 

        for i in range(len(controller_position_setpoints)):
            if control_rate is not None:
                control_rate.sleep()

            if control_function is not None and fifo_in is not None:
                control_function(controller_position_setpoints[i], fifo_in)
                
        if fifo_in is not None:
            os.close(fifo_in)

        if control_rate is not None:
            return control_rate.last_time
        else:
            return None

    def _process_action_outcome(self, base_info, action_info, robot_stopped=False):

        reward, reward_info = self._get_reward()

        if self._spline_speed_range is not None and self._spline_random_speed_per_time_step \
                and not self._spline_speed_fixed:
            self._spline_speed = np.random.uniform(self._spline_speed_range[0],
                                                   self._spline_speed_range[1])

        if self._robot_scene.floating_robot_base:
            robot_base_pos_deviation = self._robot_scene.robot_base_pos_deviation
            robot_base_orn_deviation = self._robot_scene.robot_base_orn_deviation
            robot_z_angle_deviation_rad = self._robot_scene.robot_z_angle_deviation_rad

            for key in ['average', 'max']:
                base_info[key]['robot_base_pos_deviation'] = robot_base_pos_deviation
                base_info[key]['robot_base_orn_deviation'] = robot_base_orn_deviation
                base_info[key]['robot_z_angle_deviation_rad'] = robot_z_angle_deviation_rad

        observation, observation_info = self._get_observation()
        done = self._check_termination(robot_stopped)

        if self._use_splines and self._termination_reason == self.TERMINATION_ROBOT_STOPPED:
            # multiply the resulting reward to ensure that an early termination does not lead to a lower reward
            reward = (self._trajectory_manager.trajectory_length - self._episode_length) * reward

        info = defaultdict(dict)

        for k, v in chain(base_info.items(), action_info.items(), observation_info.items(), reward_info.items()):
            info[k] = {**info[k], **v}

        return observation, reward, done, info

    def _process_end_of_episode(self, observation, reward, done, info):
        if self._trajectory_successful:
            info['trajectory_successful'] = 1.0
        else:
            info['trajectory_successful'] = 0.0

        info.update(trajectory_length=self._trajectory_manager.trajectory_length)
        info.update(episode_length=self._episode_length)
        info['termination_reason'] = self._termination_reason

        logging.info("Termination reason: %s", self._termination_reason)

        # get info from obstacle wrapper
        obstacle_info = self._robot_scene.obstacle_wrapper.get_info_and_print_stats()
        info = dict(info, **obstacle_info)  # concatenate dicts
        self._display_plot()
        self._save_plot(self.__class__.__name__, self._experiment_name)

        return observation, reward, done, info

    def _store_action_list(self):
        action_dict = {'actions': np.asarray(self._action_list).tolist()}
        eval_dir = os.path.join(self._evaluation_dir, "action_logs")

        if not os.path.exists(eval_dir):
            try:
                os.makedirs(eval_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        with open(os.path.join(eval_dir, "episode_{}_{}.json".format(self._episode_counter, self.pid)), 'w') as f:
            f.write(json.dumps(action_dict))
            f.flush()

    def _store_trajectory_data(self):
        trajectory_dict = {'actions': np.asarray(self._action_list).tolist(),
                           'trajectory_setpoints': self._to_list(
                               self._trajectory_manager.generated_trajectory_control_points),
                           'trajectory_measured_actual_values': self._to_list(
                               self._trajectory_manager.measured_actual_trajectory_control_points),
                           'trajectory_computed_actual_values': self._to_list(
                               self._trajectory_manager.computed_actual_trajectory_control_points),
                           }
        eval_dir = os.path.join(self._evaluation_dir, "trajectory_data")

        if not os.path.exists(eval_dir):
            try:
                os.makedirs(eval_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        with open(os.path.join(eval_dir, "episode_{}_{}.json".format(self._episode_counter, self.pid)), 'w') as f:
            f.write(json.dumps(trajectory_dict))
            f.flush()

    def close(self):
        self._robot_scene.disconnect()
        self._disconnect_physic_clients()
        if self._movement_process_pool is not None:
            self._movement_process_pool.close()
            self._movement_process_pool.join()

    def _disconnect_physic_clients(self):
        if self._egl_plugin is not None:
            p.unloadPlugin(self._egl_plugin)
        for i in range(self._num_physic_clients):
            p.disconnect(physicsClientId=i)

    def set_seed(self, seed=None):
        self._fixed_seed = seed
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    @staticmethod
    def _to_list(dictionary):
        for key, value in dictionary.items():
            dictionary[key] = np.asarray(value).tolist()
        return dictionary

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError()

    @abstractmethod
    def _get_observation(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_reward(self):
        raise NotImplementedError()

    @abstractmethod
    def _display_plot(self):
        raise NotImplementedError()

    @abstractmethod
    def _save_plot(self, class_name, experiment_name):
        raise NotImplementedError()

    @abstractmethod
    def acc_braking_function(self):
        raise NotImplementedError()

    def compute_next_acc_min_and_next_acc_max(self, start_position, start_velocity, start_acceleration):
        raise NotImplementedError()

    @abstractmethod
    def _compute_controller_setpoints_from_action(self, action):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_position(self, step):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_velocity(self, step):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_acceleration(self, step):
        raise NotImplementedError()

    def _sim_step(self):
        p.stepSimulation(physicsClientId=self._simulation_client_id)
        self._control_step_counter += 1

    def _prepare_for_next_action(self):
        return

    def _prepare_for_end_of_episode(self):
        return

    def _check_termination(self, robot_stopped):
        done, termination_reason = \
            self._trajectory_manager.is_trajectory_finished(self._current_trajectory_point_index)
        if done and self._termination_reason == self.TERMINATION_UNSET:
            self._termination_reason = termination_reason

        if not done and robot_stopped and self._terminate_on_robot_stop:
            done = True
            self._termination_reason = self.TERMINATION_ROBOT_STOPPED

        if not done and self._robot_scene.sphere_balancing_mode and (self._terminate_on_balancing_sphere_deviation
                                                                     or self._terminate_balancing_sphere_not_on_board):
            for i in range(self._robot_scene.num_robots):
                sphere_pos_local, sphere_pos_local_rel = self._robot_scene.get_sphere_position_on_board(robot=i)

                if self._terminate_balancing_sphere_not_on_board:
                    # check if sphere is on board
                    if not self._robot_scene.is_sphere_on_board(sphere_pos_local_rel):
                        done = True
                        self._termination_reason = self.TERMINATION_BALANCING
                if self._terminate_on_balancing_sphere_deviation and self._balancing_sphere_dev_min_max is not None:
                    # check if sphere is away from the initial position
                    sphere_distance = np.linalg.norm(
                        np.array(sphere_pos_local) - np.array(self._robot_scene.sphere_start_pos_list[i]))
                    if sphere_distance > self._balancing_sphere_dev_min_max[1]:
                        done = True
                        self._termination_reason = self.TERMINATION_BALANCING

        if self._robot_scene.robot_base_balancing_mode:
            if not done and self._terminate_on_balancing_robot_base_pos_deviation:
                if self._robot_scene.robot_base_pos_deviation > self._balancing_robot_base_max_pos_deviation:
                    done = True
                    self._termination_reason = self.TERMINATION_BASE_POS_DEVIATION

            if not done and self._terminate_on_balancing_robot_base_orn_deviation:
                if self._robot_scene.robot_base_orn_deviation > self._balancing_robot_base_max_orn_deviation:
                    done = True
                    self._termination_reason = self.TERMINATION_BASE_ORN_DEVIATION

            if not done and self._terminate_on_balancing_robot_base_z_angle_deviation:
                if self._robot_scene.robot_z_angle_deviation_rad > self._balancing_robot_base_max_z_angle_deviation_rad:
                    done = True
                    self._termination_reason = self.TERMINATION_BASE_Z_ANGLE_DEVIATION

        return done

    @property
    def trajectory_time_step(self):
        return self._trajectory_time_step

    @property
    def pid(self):
        return self._pid

    @property
    def evaluation_dir(self):
        return self._evaluation_dir

    @property
    def use_real_robot(self):
        return self._use_real_robot

    @property
    def episode_counter(self):
        return self._episode_counter

    @property
    def precomputation_time(self):
        if self._control_rate is not None and hasattr(self._control_rate, 'precomputation_time'):
            return self._control_rate.precomputation_time
        else:
            return None

    @property
    def use_splines(self):
        return self._use_splines

    @property
    def sphere_balancing_mode(self):
        return self._sphere_balancing_mode

    @property
    def floating_robot_base(self):
        return self._floating_robot_base

    @property
    @abstractmethod
    def pos_limits_min_max(self):
        pass

    @abstractmethod
    def _get_safe_acc_range(self):
        pass

    @abstractmethod
    def _reset_plotter(self, initial_joint_position):
        pass

    @abstractmethod
    def _add_actual_position_to_plot(self, actual_position):
        pass

    @abstractmethod
    def _add_computed_actual_position_to_plot(self, computed_position_is, computed_velocity_is,
                                              computed_acceleration_is):
        pass

    @abstractmethod
    def _add_baseline_position_to_plot(self, baseline_position_is, baseline_velocity_is, baseline_acceleration_is):
        pass

    @abstractmethod
    def _add_actual_torques_to_plot(self, actual_torques):
        pass

    @abstractmethod
    def _calculate_safe_acc_range(self, start_position, start_velocity, start_acceleration, trajectory_point_index):
        pass

    def _get_trajectory_start_position(self):
        return self._trajectory_manager.get_trajectory_start_position()

    def _get_generated_trajectory_point(self, index, key='positions'):
        return self._trajectory_manager.get_generated_trajectory_point(index, key)

    def _get_measured_actual_trajectory_control_point(self, index, start_at_index=False, key='positions'):
        return self._trajectory_manager.get_measured_actual_trajectory_control_point(index, start_at_index, key)

    def _get_computed_actual_trajectory_control_point(self, index, start_at_index=False, key='positions'):
        return self._trajectory_manager.get_computed_actual_trajectory_control_point(index, start_at_index, key)

    def _get_generated_trajectory_control_point(self, index, key='positions'):
        return self._trajectory_manager.get_generated_trajectory_control_point(index, key)

    def _add_generated_trajectory_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_generated_trajectory_point(position, velocity, acceleration)

    def _add_measured_actual_trajectory_control_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_measured_actual_trajectory_control_point(position, velocity, acceleration)

    def _add_computed_actual_trajectory_control_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_computed_actual_trajectory_control_point(position, velocity, acceleration)

    def _add_generated_trajectory_control_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_generated_trajectory_control_point(position, velocity, acceleration)


def normalize_joint_values(values, joint_limits):
    return list(np.asarray(values) / np.asarray(joint_limits))
