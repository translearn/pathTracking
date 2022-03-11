# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
from abc import ABC

import numpy as np
from gym.spaces import Box
from klimits import normalize as normalize_array
from tracking.envs.tracking_base import TrackingBase
from tracking.utils.spline_utils import Spline


def normalize_joint_values(values, joint_limits):
    return list(np.array(values) / np.array(joint_limits))


def normalize(value, min_value, max_value):
    return -1 + 2 * (value - min_value) / (max_value - min_value)


def _normalize_joint_values_min_max(values, joint_limit_ranges):
    normalized_values = -1 + 2 * (values - joint_limit_ranges[0]) / (joint_limit_ranges[1] - joint_limit_ranges[0])
    continuous_joint_indices = np.isnan(joint_limit_ranges[0]) | np.isnan(joint_limit_ranges[1])
    if np.any(continuous_joint_indices):
        # continuous joint -> map [-np.pi, np.pi] and all values shifted by 2 * np.pi to [-1, 1]
        normalized_values[continuous_joint_indices] = \
            -1 + 2 * (((values[continuous_joint_indices] + np.pi)/(2 * np.pi)) % 1)
    return normalized_values


TARGET_POINT_SIMULTANEOUS = 0
TARGET_POINT_ALTERNATING = 1
TARGET_POINT_SINGLE = 2


class TrackingObservation(ABC, TrackingBase):

    def __init__(self,
                 *vargs,
                 m_prev=0,
                 obs_spline_n_next=5,
                 obs_spline_add_length=False,
                 obs_spline_add_distance_per_knot=False,
                 obs_spline_use_distance_between_knots=False,
                 obs_spline_visualization=True,
                 obs_no_balancing_sphere=False,
                 **kwargs):
        super().__init__(*vargs, **kwargs)

        self._m_prev = m_prev
        self._next_joint_acceleration_mapping = None

        self._obs_spline_n_next = obs_spline_n_next
        self._obs_spline_add_length = obs_spline_add_length
        self._obs_spline_add_distance_per_knot = obs_spline_add_distance_per_knot
        self._obs_spline_use_distance_between_knots = obs_spline_use_distance_between_knots

        obs_current_size = 3  # pos, vel, acc

        obs_target_point_size = 0

        obs_spline_size = 0
        self._obs_spline_visualization = obs_spline_visualization
        self._obs_spline_debug_line_buffer = None

        self._obs_no_balancing_sphere = obs_no_balancing_sphere

        obs_sphere_balancing_size = 0
        obs_robot_base_balancing_size = 0

        if self._robot_scene.obstacle_wrapper.use_target_points:
            if self._obs_add_target_point_pos:
                if self._robot_scene.obstacle_wrapper.target_point_sequence == TARGET_POINT_SINGLE:
                    obs_target_point_size += 3  # one target point only
                else:
                    obs_target_point_size += 3 * self._robot_scene.num_robots  # one target point per robot

            if self._obs_add_target_point_relative_pos:
                obs_target_point_size += 3 * self._robot_scene.num_robots

            if self._robot_scene.obstacle_wrapper.target_point_sequence == TARGET_POINT_ALTERNATING:
                obs_target_point_size += self._robot_scene.num_robots  # target point active signal for each robot

        if self._use_splines:
            # reference_spline next knots, current knot (=first relevant knot) is always added
            obs_spline_size += np.sum(self._robot_scene.spline_joint_mask) * (self._obs_spline_n_next + 1)
            obs_spline_size += 1
            # reference_spline relative start in [-1, 1] -> indicates the position of reference_spline.current_length
            # relative to the length of the curve between the first and the second relevant reference_spline knot

            if self._obs_spline_add_length:
                obs_spline_size += 1
                # reference_spline relative end in [-1, 1] -> indicates the relative position of the last knot of the
                # reference spline relative to the maximal curve length resulting
                # from self._obs_spline_n_next * spline_max_distance_between_knots

            if self._obs_spline_add_distance_per_knot:
                obs_spline_size += self._obs_spline_n_next
                # distance per knot, either the distance from the first knot relative to
                # self._obs_spline_n_next * spline_max_distance_between_knots
                # (self._obs_spline_use_distance_between_knots == False)
                # or the distance between each knot
                # relative to spline_max_distance_between_knots (self._obs_spline_use_distance_between_knots == True)

            if self._spline_speed_range is not None:
                obs_spline_size += 1  # optional speed signal in [-1, 1]

            if self._obs_add_target_point_pos:
                # cartesian position according to the current (starting) point of the reference spline
                obs_spline_size += 3 * self._robot_scene.num_robots

            if self._obs_add_target_point_relative_pos:
                # relative cartesian distance between the robot and the current (starting) point of the reference spline
                obs_spline_size += 3 * self._robot_scene.num_robots

        if self._robot_scene.sphere_balancing_mode and not self._obs_no_balancing_sphere:
            obs_sphere_balancing_size = 4  # last sphere pos (x, y) and current sphere pos (x, y)
            if self._balancing_sphere_dev_min_max is not None:
                obs_sphere_balancing_size += 2  # deviation from the sphere starting position (delta x, delta y)

        if self._robot_scene.robot_base_balancing_mode:
            obs_robot_base_balancing_size += 3  # base pos [x, y, z]
            obs_robot_base_balancing_size += 4  # base orn quaternion [a, b, c, d]
            obs_robot_base_balancing_size += 1  # z angle deviation

        self._last_sphere_pos_board_list = None

        self._observation_size = self._m_prev * self._num_manip_joints \
            + obs_current_size * self._num_manip_joints \
            + obs_target_point_size + obs_spline_size + obs_sphere_balancing_size * self._robot_scene.num_robots \
            + obs_robot_base_balancing_size

        self.observation_space = Box(low=np.float32(-1), high=np.float32(1), shape=(self._observation_size,),
                                     dtype=np.float32)

        logging.info("Observation size: " + str(self._observation_size))

    def reset(self, **kwargs):
        if self._use_gui and self._obs_spline_visualization:
            # reset obs spline for video rendering with imagegrab
            self._obs_spline_debug_line_buffer = \
                Spline.reset_debug_line_buffer(self._obs_spline_debug_line_buffer,
                                               physics_client_id=self._gui_client_id)

        super().reset(**kwargs)
        self._robot_scene.prepare_for_start_of_episode()
        self._last_sphere_pos_board_list = None

        observation, observation_info = self._get_observation()

        if self._control_rate is not None and hasattr(self._control_rate, 'reset'):
            # reset control rate timer
            self._control_rate.reset()

        return observation

    def _prepare_for_next_action(self):
        super()._prepare_for_next_action()

    def _get_observation(self):
        prev_joint_accelerations = self._get_m_prev_joint_values(self._m_prev, key='accelerations')
        curr_joint_position = self._get_generated_trajectory_point(-1)
        curr_joint_velocity = self._get_generated_trajectory_point(-1, key='velocities')
        curr_joint_acceleration = self._get_generated_trajectory_point(-1, key='accelerations')
        prev_joint_accelerations_rel = [normalize_joint_values(p, self._robot_scene.max_accelerations)
                                        for p in prev_joint_accelerations]
        curr_joint_position_rel_obs = list(_normalize_joint_values_min_max(curr_joint_position,
                                                                           self.pos_limits_min_max))
        curr_joint_velocity_rel_obs = normalize_joint_values(curr_joint_velocity, self._robot_scene.max_velocities)
        curr_joint_acceleration_rel_obs = normalize_joint_values(curr_joint_acceleration,
                                                                 self._robot_scene.max_accelerations)

        # target point for reaching tasks
        target_point_rel_obs = []
        if self._robot_scene.obstacle_wrapper.use_target_points:
            # the function needs to be called even if the return value is not used.
            # otherwise, new target points are not generated
            target_point_pos, target_point_relative_pos, _, target_point_active_obs = \
                self._robot_scene.obstacle_wrapper.get_target_point_observation(
                    compute_relative_pos_norm=self._obs_add_target_point_relative_pos,
                    compute_target_point_joint_pos_norm=False)
            if self._obs_add_target_point_pos:
                target_point_rel_obs = target_point_rel_obs + list(target_point_pos)

            if self._obs_add_target_point_relative_pos:
                target_point_rel_obs = target_point_rel_obs + list(target_point_relative_pos)

            target_point_rel_obs = target_point_rel_obs + list(target_point_active_obs)
            # to indicate if the target point is active (1.0) or inactive (0.0); the list is empty if not required

        spline_rel_obs = []
        if self._use_splines:
            # add knot positions, current knot (=first relevant knot) is always added
            current_knots_curve_data, current_knots_dis, current_knots_indices = \
                self._trajectory_manager.reference_spline.get_current_knots(n_next=self._obs_spline_n_next)

            self._trajectory_manager.reference_spline_last_obs_dis = current_knots_dis[-1]  # for reward calculation
            # normalize curve_data (=joint positions)
            spline_knots_pos_rel = \
                _normalize_joint_values_min_max(current_knots_curve_data,
                                                self.pos_limits_min_max[:, self._robot_scene.spline_joint_mask])
            spline_knots_pos_rel_obs = list(spline_knots_pos_rel.ravel())

            # compute normalized start distance
            spline_start_rel_obs = [normalize(value=self._trajectory_manager.reference_spline.current_dis -
                                              current_knots_dis[0],
                                              min_value=0,
                                              max_value=self._trajectory_manager.spline_max_distance_between_knots)]

            if self._obs_spline_add_length:
                # reference_spline relative end in [-1, 1] -> indicates the relative position of the last knot of the
                # reference spline relative to the maximal curve length resulting
                # from self._obs_spline_n_next * spline_max_distance_between_knots
                spline_end_rel_obs = [normalize(value=current_knots_dis[-1] - current_knots_dis[0],
                                                min_value=0,
                                                max_value=self._obs_spline_n_next *
                                                self._trajectory_manager.spline_max_distance_between_knots)]
            else:
                spline_end_rel_obs = []

            if self._obs_spline_add_distance_per_knot:
                # distance per knot
                if self._obs_spline_use_distance_between_knots:
                    # distance between each knot relative to spline_max_distance_between_knots
                    distance_between_knots = np.diff(current_knots_dis)
                    spline_knot_distance_rel_obs = \
                        list(normalize(value=distance_between_knots,
                                       min_value=0,
                                       max_value=self._trajectory_manager.spline_max_distance_between_knots))
                else:
                    # distance from the first knot relative to
                    # self._obs_spline_n_next * spline_max_distance_between_knots
                    distance_from_first_knot = current_knots_dis[1:] - current_knots_dis[0]
                    spline_knot_distance_rel_obs =\
                        list(normalize(value=distance_from_first_knot,
                                       min_value=0,
                                       max_value=
                                       self._obs_spline_n_next *
                                       self._trajectory_manager.spline_max_distance_between_knots))
            else:
                spline_knot_distance_rel_obs = []

            if self._spline_speed_range is not None:
                spline_speed_rel_obs = [normalize(value=self._spline_speed,
                                                  min_value=self._spline_speed_range[0],
                                                  max_value=self._spline_speed_range[1])]
            else:
                spline_speed_rel_obs = []

            spline_target_point_rel_obs = []
            if self._obs_add_target_point_pos:
                for i in range(self._robot_scene.num_robots):
                    spline_target_point_rel_obs += \
                        list(normalize_array(self._trajectory_manager.spline_target_point[i][0:3],
                                             self._robot_scene.obstacle_wrapper.target_point_cartesian_range_min_max))

            if self._obs_add_target_point_relative_pos:
                for i in range(self._robot_scene.num_robots):
                    for j in range(3):
                        spline_target_point_rel_obs.append(normalize(
                            value=self._trajectory_manager.spline_target_point_relative_pos[i, j],
                            min_value=-self._spline_cartesian_deviation_max_threshold,
                            max_value=self._spline_cartesian_deviation_max_threshold))

            spline_rel_obs = spline_knots_pos_rel_obs + spline_start_rel_obs \
                + spline_end_rel_obs + spline_knot_distance_rel_obs + spline_speed_rel_obs + spline_target_point_rel_obs

            if self._use_gui and self._obs_spline_visualization:
                u_start = self._trajectory_manager.reference_spline.u[current_knots_indices[0]]
                u_end = self._trajectory_manager.reference_spline.u[current_knots_indices[-1]]
                u_marker = self._trajectory_manager.reference_spline.length_to_u(
                    length=self._trajectory_manager.reference_spline.current_dis)
                self._obs_spline_debug_line_buffer = \
                    self._trajectory_manager.reference_spline.visualize(env=self,
                                                                        debug_line_buffer=
                                                                        self._obs_spline_debug_line_buffer,
                                                                        visualize_knots=True,
                                                                        visualize_knots_orn=False,
                                                                        sample_distance=0.05,
                                                                        use_normalized_sample_distance=False,
                                                                        debug_line_buffer_init_size=80, u_start=u_start,
                                                                        u_end=u_end,
                                                                        line_color=[0, 0, 1],
                                                                        line_width=4,
                                                                        u_marker=u_marker,
                                                                        marker_color=[1, 0, 0],
                                                                        marker_width=3,
                                                                        physics_client_id=self._gui_client_id)

        balancing_sphere_rel_obs = []

        if self._robot_scene.sphere_balancing_mode and not self._obs_no_balancing_sphere:
            curr_sphere_pos_board_list = [self._robot_scene.get_sphere_position_on_board(robot=i)
                                          for i in range(self._robot_scene.num_robots)]

            if self._last_sphere_pos_board_list is None:
                self._last_sphere_pos_board_list = curr_sphere_pos_board_list

            for i in range(self._robot_scene.num_robots):
                # last sphere pos rel (x, y)
                balancing_sphere_rel_obs += self._last_sphere_pos_board_list[i][1][:2]
                # current sphere pos rel (x, y)
                balancing_sphere_rel_obs += curr_sphere_pos_board_list[i][1][:2]

                if self._balancing_sphere_dev_min_max is not None:
                    # deviation from the sphere starting position (delta x, delta y)
                    # divided by two to ensure range [-1, 1]
                    balancing_sphere_rel_obs += list((np.array(curr_sphere_pos_board_list[i][1][:2]) -
                                                      np.array(self._robot_scene.sphere_start_pos_rel_list[i][:2])) / 2)

            self._last_sphere_pos_board_list = curr_sphere_pos_board_list

        balancing_robot_base_rel_obs = []

        if self._robot_scene.robot_base_balancing_mode:
            # base pos [x, y, z] with x, y, z in [-robot_base_pos_max_value, robot_base_pos_max_value]
            # normalized to [-1, 1]
            robot_base_pos_max_value = self._balancing_robot_base_max_pos_deviation / np.sqrt(3)
            balancing_robot_base_rel_obs += list(normalize(value=self._robot_scene.actual_base_pos,
                                                           min_value=-robot_base_pos_max_value,
                                                           max_value=robot_base_pos_max_value))
            # base orn [a, b, c, d] with a, b, c, d in [0, 1] -> unit quaternion
            balancing_robot_base_rel_obs += list(normalize(value=self._robot_scene.actual_base_orn,
                                                           min_value=0,
                                                           max_value=1))
            # z_angle_deviation in [0, self._balancing_robot_base_max_z_angle_deviation_rad]
            balancing_robot_base_rel_obs += [normalize(value=self._robot_scene.robot_z_angle_deviation_rad,
                                                       min_value=0,
                                                       max_value=self._balancing_robot_base_max_z_angle_deviation_rad
                                                       )]

        observation = np.array((np.core.umath.clip(
            [item for sublist in prev_joint_accelerations_rel for item in sublist]
            + curr_joint_position_rel_obs + curr_joint_velocity_rel_obs
            + curr_joint_acceleration_rel_obs
            + target_point_rel_obs
            + spline_rel_obs
            + balancing_sphere_rel_obs
            + balancing_robot_base_rel_obs, -1, 1)), dtype=np.float32)

        info = {'average': {},
                'max': {},
                'min': {}}

        pos_violation = 0.0
        vel_violation = 0.0
        acc_violation = 0.0

        for j in range(self._num_manip_joints):

            info['average']['joint_{}_pos'.format(j)] = curr_joint_position_rel_obs[j]
            info['average']['joint_{}_pos_abs'.format(j)] = abs(curr_joint_position_rel_obs[j])
            info['max']['joint_{}_pos'.format(j)] = curr_joint_position_rel_obs[j]
            info['min']['joint_{}_pos'.format(j)] = curr_joint_position_rel_obs[j]
            if abs(curr_joint_position_rel_obs[j]) > 1.001:
                logging.warning("Position violation: t = %s Joint: %s Rel position %s",
                                (self._episode_length - 1) * self._trajectory_time_step, j,
                                curr_joint_position_rel_obs[j])
                pos_violation = 1.0

            info['average']['joint_{}_vel'.format(j)] = curr_joint_velocity_rel_obs[j]
            info['average']['joint_{}_vel_abs'.format(j)] = abs(curr_joint_velocity_rel_obs[j])
            info['max']['joint_{}_vel'.format(j)] = curr_joint_velocity_rel_obs[j]
            info['min']['joint_{}_vel'.format(j)] = curr_joint_velocity_rel_obs[j]
            if abs(curr_joint_velocity_rel_obs[j]) > 1.001:
                logging.warning("Velocity violation: t = %s Joint: %s Rel velocity %s",
                                (self._episode_length - 1) * self._trajectory_time_step, j,
                                curr_joint_velocity_rel_obs[j])
                vel_violation = 1.0

            info['average']['joint_{}_acc'.format(j)] = curr_joint_acceleration_rel_obs[j]
            info['average']['joint_{}_acc_abs'.format(j)] = abs(curr_joint_acceleration_rel_obs[j])
            info['max']['joint_{}_acc'.format(j)] = curr_joint_acceleration_rel_obs[j]
            info['min']['joint_{}_acc'.format(j)] = curr_joint_acceleration_rel_obs[j]
            if abs(curr_joint_acceleration_rel_obs[j]) > 1.001:
                logging.warning("Acceleration violation: t = %s Joint: %s Rel acceleration %s",
                                (self._episode_length - 1) * self._trajectory_time_step, j,
                                curr_joint_acceleration_rel_obs[j])
                acc_violation = 1.0

        info['average']['joint_vel_norm'] = np.linalg.norm(curr_joint_velocity_rel_obs)
        info['max']['joint_pos_violation'] = pos_violation
        info['max']['joint_vel_violation'] = vel_violation
        info['max']['joint_acc_violation'] = acc_violation

        logging.debug("Observation %s: %s", self._episode_length, np.asarray(observation))

        return observation, info

    def _get_m_prev_joint_values(self, m, key):

        m_prev_joint_values = []

        for i in range(m+1, 1, -1):
            m_prev_joint_values.append(self._get_generated_trajectory_point(-i, key))

        return m_prev_joint_values
