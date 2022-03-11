# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import logging
from abc import ABC

import numpy as np

from tracking.envs.tracking_base import TrackingBase


def normalize_joint_values(values, joint_limits):
    return list(np.array(values) / np.array(joint_limits))


def compute_quadratic_punishment(a, b, c, d):
    # returns max(min((a - b) / (c - d), 1), 0) ** 2
    punishment = (a - b) / (c - d)
    return max(min(punishment, 1), 0) ** 2


class RewardBase(ABC, TrackingBase):
    # optional action penalty
    ACTION_THRESHOLD = 0.9
    ACTION_MAX_PUNISHMENT = 1.0

    ACTION_REWARD_PEAK = 0.95
    ACTION_MAX_REWARD = 1.0

    def __init__(self,
                 *vargs,
                 normalize_reward_to_frequency=False,
                 punish_action=False,
                 action_punishment_min_threshold=ACTION_THRESHOLD,
                 action_max_punishment=ACTION_MAX_PUNISHMENT,
                 reward_action=False,
                 action_reward_min_threshold=ACTION_THRESHOLD,
                 action_reward_peak=ACTION_REWARD_PEAK,
                 action_max_reward=ACTION_MAX_REWARD,
                 **kwargs):
        # reward settings
        self.reward_range = [0, 1]  # dummy settings
        self._normalize_reward_to_frequency = normalize_reward_to_frequency

        self._punish_action = punish_action
        self._action_punishment_min_threshold = action_punishment_min_threshold
        self._action_max_punishment = action_max_punishment

        self._reward_action = reward_action
        self._action_reward_min_threshold = action_reward_min_threshold
        self._action_reward_peak = action_reward_peak
        self._action_max_reward = action_max_reward

        super().__init__(*vargs, **kwargs)

    def _compute_action_punishment(self):
        # The aim of the action punishment is to avoid the action being too close to -1 or 1.
        action_abs = np.abs(self._last_action)
        max_action_abs = max(action_abs)
        return compute_quadratic_punishment(max_action_abs, self._action_punishment_min_threshold,
                                            1, self._action_punishment_min_threshold)

    def _compute_action_reward(self):
        # The aim of the action reward is to encourage (at least) one action to be close to 1 / -1
        # -> time-optimized motion
        # reward:
        # max_action_abs <= action_reward_min_threshold -> 0.0
        # max_action_abs == action_reward_peak -> 1.0
        # max_action_abs == 1.0 -> - 1 * ((1.0 - self._action_reward_peak) /
        #                                    (self._action_reward_min_threshold - self._action_reward_peak))**2 + 1.0
        action_abs = np.abs(self._last_action)
        max_action_abs = max(action_abs)
        if max_action_abs > self._action_reward_min_threshold:
            action_reward = - 1 * ((max_action_abs - self._action_reward_peak) /
                                   (self._action_reward_min_threshold - self._action_reward_peak))**2 + 1.0
            return max(action_reward, 0)
        else:
            return 0.0

    def _common_reward_function(self, reward, info):
        if self._normalize_reward_to_frequency:
            # Baseline: 10 Hz
            reward = reward * self._trajectory_time_step / 0.1

        for key in ['average', 'min', 'max']:
            info[key].update(reward=reward)

        # add information about the jerk as custom metric
        curr_joint_jerk = \
            (np.array(self._get_generated_trajectory_point(-1, key='accelerations'))
             - np.array(self._get_generated_trajectory_point(-2, key='accelerations'))) \
            / self._trajectory_time_step

        curr_joint_jerk_rel = normalize_joint_values(curr_joint_jerk, self._robot_scene.max_jerk_linear_interpolation)
        jerk_violation = 0.0

        for j in range(self._num_manip_joints):
            info['average']['joint_{}_jerk'.format(j)] = curr_joint_jerk_rel[j]
            info['average']['joint_{}_jerk_abs'.format(j)] = abs(curr_joint_jerk_rel[j])
            info['max']['joint_{}_jerk'.format(j)] = curr_joint_jerk_rel[j]
            info['min']['joint_{}_jerk'.format(j)] = curr_joint_jerk_rel[j]

        max_normalized_jerk = np.max(np.abs(curr_joint_jerk_rel))
        if max_normalized_jerk > 1.002:
            jerk_violation = 1.0
            logging.warning("Jerk violation: t = %s Joint: %s Rel jerk %s",
                            (self._episode_length - 1) * self._trajectory_time_step,
                            np.argmax(np.abs(curr_joint_jerk_rel)),
                            max_normalized_jerk)

        info['max']['joint_jerk_violation'] = jerk_violation

        logging.debug("Reward %s: %s", self._episode_length - 1, reward)

        return reward, info

    @property
    def reward_maximum_relevant_distance(self):
        return None


class TargetPointReachingReward(RewardBase):
    ADAPTATION_MAX_PUNISHMENT = 1.0
    END_MIN_DISTANCE_MAX_PUNISHMENT = 1.0
    END_MAX_TORQUE_MAX_PUNISHMENT = 1.0
    END_MAX_TORQUE_MIN_THRESHOLD = 0.9

    # braking trajectory max punishment (either collision or torque -> max)
    BRAKING_TRAJECTORY_MAX_PUNISHMENT = 1.0
    # braking trajectory torque penalty
    BRAKING_TRAJECTORY_MAX_TORQUE_MIN_THRESHOLD = 0.9  # rel. abs. torque threshold

    # reference_spline settings
    SPLINE_MAX_DEVIATION_MAX_PUNISHMENT = 1.0

    def __init__(self,
                 *vargs,
                 normalize_reward_to_initial_target_point_distance=False,
                 punish_adaptation=False,
                 adaptation_max_punishment=ADAPTATION_MAX_PUNISHMENT,
                 punish_end_min_distance=False,
                 end_min_distance_max_punishment=END_MIN_DISTANCE_MAX_PUNISHMENT,
                 end_min_distance_max_threshold=None,
                 punish_end_max_torque=False,
                 end_max_torque_max_punishment=END_MAX_TORQUE_MAX_PUNISHMENT,
                 end_max_torque_min_threshold=END_MAX_TORQUE_MIN_THRESHOLD,
                 braking_trajectory_max_punishment=BRAKING_TRAJECTORY_MAX_PUNISHMENT,
                 punish_braking_trajectory_min_distance=False,
                 braking_trajectory_min_distance_max_threshold=None,
                 punish_braking_trajectory_max_torque=False,
                 braking_trajectory_max_torque_min_threshold=BRAKING_TRAJECTORY_MAX_TORQUE_MIN_THRESHOLD,
                 target_point_reward_factor=1.0,
                 **kwargs):

        self._normalize_reward_to_initial_target_point_distance = normalize_reward_to_initial_target_point_distance

        self._punish_adaptation = punish_adaptation
        self._adaptation_max_punishment = adaptation_max_punishment

        self._punish_end_min_distance = punish_end_min_distance
        self._end_min_distance_max_punishment = end_min_distance_max_punishment
        self._end_min_distance_max_threshold = end_min_distance_max_threshold
        self._punish_end_max_torque = punish_end_max_torque
        self._end_max_torque_max_punishment = end_max_torque_max_punishment
        self._end_max_torque_min_threshold = end_max_torque_min_threshold

        self._punish_braking_trajectory_min_distance = punish_braking_trajectory_min_distance
        self._braking_trajectory_min_distance_max_threshold = braking_trajectory_min_distance_max_threshold
        self._punish_braking_trajectory_max_torque = punish_braking_trajectory_max_torque
        self._braking_trajectory_max_punishment = braking_trajectory_max_punishment
        self._max_torque_min_threshold = braking_trajectory_max_torque_min_threshold

        self._target_point_reward_factor = target_point_reward_factor
        self._reward_maximum_relevant_distance = None

        if self._punish_braking_trajectory_min_distance or self._punish_end_min_distance:
            if self._punish_braking_trajectory_min_distance and \
                    self._braking_trajectory_min_distance_max_threshold is None:
                raise ValueError("punish_braking_trajectory_min_distance requires "
                                 "braking_trajectory_min_distance_max_threshold to be specified")
            if self._punish_end_min_distance and \
                    self._end_min_distance_max_threshold is None:
                raise ValueError("punish_end_min_distance requires "
                                 "end_min_distance_max_threshold to be specified")

            if self._punish_braking_trajectory_min_distance and self._punish_end_min_distance:
                self._reward_maximum_relevant_distance = max(self._braking_trajectory_min_distance_max_threshold,
                                                             self._end_min_distance_max_threshold)
            elif self._punish_braking_trajectory_min_distance:
                self._reward_maximum_relevant_distance = self._braking_trajectory_min_distance_max_threshold
            else:
                self._reward_maximum_relevant_distance = self._end_min_distance_max_threshold

        super().__init__(*vargs, **kwargs)

    def _get_reward(self):
        info = {'average': {}, 'min': {}, 'max': {}}

        reward = 0
        target_point_reward = 0
        action_punishment = 0
        adaptation_punishment = 0
        end_min_distance_punishment = 0
        end_max_torque_punishment = 0

        braking_trajectory_min_distance_punishment = 0
        braking_trajectory_max_torque_punishment = 0

        if self._robot_scene.obstacle_wrapper.use_target_points:
            if self._punish_action:
                action_punishment = self._compute_action_punishment()  # action punishment

            if self._punish_adaptation:
                adaptation_punishment = self._adaptation_punishment

            if self._punish_end_min_distance:
                if self._end_min_distance is None:
                    self._end_min_distance, _, _, _ = self._robot_scene.obstacle_wrapper.get_minimum_distance(
                        manip_joint_indices=self._robot_scene.manip_joint_indices,
                        target_position=self._start_position)

                end_min_distance_punishment = compute_quadratic_punishment(
                    a=self._end_min_distance_max_threshold,
                    b=self._end_min_distance,
                    c=self._end_min_distance_max_threshold,
                    d=self._robot_scene.obstacle_wrapper.closest_point_safety_distance)

            if self._punish_end_max_torque:
                if self._end_max_torque is not None:
                    # None if check_braking_trajectory is False and asynchronous movement execution is active
                    # in this case, no penalty is computed, but the penalty is not required anyways
                    end_max_torque_punishment = compute_quadratic_punishment(
                        a=self._end_max_torque,
                        b=self._end_max_torque_min_threshold,
                        c=1,
                        d=self._end_max_torque_min_threshold)

            target_point_reward = self._robot_scene.obstacle_wrapper.get_target_point_reward(
                normalize_distance_reward_to_initial_target_point_distance=
                self._normalize_reward_to_initial_target_point_distance)

            reward = target_point_reward * self._target_point_reward_factor \
                - action_punishment * self._action_max_punishment \
                - adaptation_punishment * self._adaptation_max_punishment \
                - end_min_distance_punishment * self._end_min_distance_max_punishment \
                - end_max_torque_punishment * self._end_max_torque_max_punishment

            if self._punish_braking_trajectory_min_distance or self._punish_braking_trajectory_max_torque:
                braking_trajectory_min_distance_punishment, braking_trajectory_max_torque_punishment = \
                    self._robot_scene.obstacle_wrapper.get_braking_trajectory_punishment(
                        minimum_distance_max_threshold=self._braking_trajectory_min_distance_max_threshold,
                        maximum_torque_min_threshold=self._max_torque_min_threshold)

                if self._punish_braking_trajectory_min_distance and self._punish_braking_trajectory_max_torque:
                    braking_trajectory_punishment = self._braking_trajectory_max_punishment * \
                                                    max(braking_trajectory_min_distance_punishment,
                                                        braking_trajectory_max_torque_punishment)
                elif self._punish_braking_trajectory_min_distance:
                    braking_trajectory_punishment = self._braking_trajectory_max_punishment * \
                                                    braking_trajectory_min_distance_punishment
                else:
                    braking_trajectory_punishment = self._braking_trajectory_max_punishment * \
                                                    braking_trajectory_max_torque_punishment

                reward = reward - braking_trajectory_punishment

        for key in ['average', 'min', 'max']:
            info[key].update(action_punishment=action_punishment,
                             adaptation_punishment=adaptation_punishment,
                             end_min_distance_punishment=end_min_distance_punishment,
                             end_max_torque_punishment=end_max_torque_punishment,
                             target_point_reward=target_point_reward,
                             braking_trajectory_min_distance_punishment=braking_trajectory_min_distance_punishment,
                             braking_trajectory_max_torque_punishment=braking_trajectory_max_torque_punishment)

        reward, info = self._common_reward_function(reward=reward, info=info)

        return reward, info

    @property
    def reward_maximum_relevant_distance(self):
        return self._reward_maximum_relevant_distance


class SplineTrackingReward(RewardBase):
    # reference_spline settings
    SPLINE_DISTANCE_MAX_REWARD = 1.0
    SPLINE_DEVIATION_MAX_THRESHOLD = 0.25
    SPLINE_MAX_DEVIATION_MAX_PUNISHMENT = 0.1
    SPLINE_MEAN_DEVIATION_MAX_PUNISHMENT = 0.1

    SPLINE_MAX_CARTESIAN_DEVIATION_MAX_PUNISHMENT = 0.1
    SPLINE_MEAN_CARTESIAN_DEVIATION_MAX_PUNISHMENT = 0.1

    BALANCING_SPHERE_MAX_REWARD = 1.0

    def __init__(self,
                 *vargs,
                 spline_distance_max_reward=SPLINE_DISTANCE_MAX_REWARD,
                 spline_deviation_max_threshold=SPLINE_DEVIATION_MAX_THRESHOLD,
                 punish_spline_max_deviation=False,
                 spline_max_deviation_max_punishment=SPLINE_MAX_DEVIATION_MAX_PUNISHMENT,
                 punish_spline_mean_deviation=False,
                 spline_mean_deviation_max_punishment=SPLINE_MEAN_DEVIATION_MAX_PUNISHMENT,
                 spline_max_cartesian_deviation_max_punishment=SPLINE_MAX_CARTESIAN_DEVIATION_MAX_PUNISHMENT,
                 spline_mean_cartesian_deviation_max_punishment=SPLINE_MEAN_CARTESIAN_DEVIATION_MAX_PUNISHMENT,
                 spline_final_overshoot_factor=1.0,
                 spline_final_distance_reward=None,
                 balancing_sphere_max_reward=BALANCING_SPHERE_MAX_REWARD,
                 balancing_robot_base_pos_max_reward=0.0,
                 balancing_robot_base_orn_max_reward=0.0,
                 balancing_robot_base_z_angle_max_reward=0.0,
                 balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold=0.4,
                 balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment=1.0,
                 **kwargs):

        self._spline_distance_max_reward = spline_distance_max_reward
        self._spline_deviation_max_threshold = spline_deviation_max_threshold
        # the maximum punishment for max_deviation_punishment and mean_deviation_punishment is reached if the
        # mean / max deviation is equal to spline_deviation_max_threshold
        self._punish_spline_max_deviation = punish_spline_max_deviation
        self._spline_max_deviation_max_punishment = spline_max_deviation_max_punishment
        self._punish_spline_mean_deviation = punish_spline_mean_deviation
        self._spline_mean_deviation_max_punishment = spline_mean_deviation_max_punishment

        self._spline_max_cartesian_deviation_max_punishment = spline_max_cartesian_deviation_max_punishment
        self._spline_mean_cartesian_deviation_max_punishment = spline_mean_cartesian_deviation_max_punishment

        self._spline_final_overshoot_factor = max(spline_final_overshoot_factor, 0.0001)
        # in (0, 1] -> smaller values reduce the distance reward in case of overshoots at the end of the spline
        self._spline_final_distance_reward = spline_final_distance_reward
        # if None: compute the distance reward at the end of the spline based on the action spline length
        # fixed value in [0, 1] -> set the distance reward at the end of the spline to a fixed value

        self._balancing_sphere_max_reward = balancing_sphere_max_reward

        self._balancing_robot_base_pos_max_reward = balancing_robot_base_pos_max_reward
        self._balancing_robot_base_orn_max_reward = balancing_robot_base_orn_max_reward
        self._balancing_robot_base_z_angle_max_reward = balancing_robot_base_z_angle_max_reward
        self._balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold = \
            balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold
        self._balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment = \
            balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment

        super().__init__(*vargs, **kwargs)

    def _get_reward(self):
        # to discourage early termination, the reward should never be negative
        # reward = distance_reward * distance_max_reward + (1 - action_punishment) * action_max_punishment \
        #    + (1 - max_deviation_punishment) * max_deviation_max_punishment \
        #    + (1 - mean_deviation_punishment) * mean_deviation_max_punishment
        # with distance_reward, action_punishment, max_deviation_punishment, mean_deviation_punishment in [0.0, 1.0]

        info = {'average': {}, 'min': {}, 'max': {}, 'sum': {}}

        reward = 0
        distance_deviation_reward = 0

        distance_reward = 0
        distance_reward_speed = 0

        action_punishment = 1.0
        action_reward = 0.0
        max_deviation_punishment = 1.0
        mean_deviation_punishment = 1.0
        max_cartesian_deviation_punishment = 1.0
        mean_cartesian_deviation_punishment = 1.0

        balancing_sphere_reward = 0

        balancing_robot_base_pos_reward = 0
        balancing_robot_base_orn_reward = 0
        balancing_robot_base_z_angle_reward = 0
        balancing_robot_base_last_cartesian_action_point_punishment = 1.0

        if self._use_splines:
            # distance reward
            # the distance reward depends on the length of the action spline, relative to the difference between
            # the distance of the last relevant knot and the current distance of the reference spline.
            # the distance reward is 1.0 if the length of the action spline is equal to this difference and decreases
            # if the action spline is shorter or longer
            # (the later case is especially relevant at the end of the reference reference_spline)

            current_to_last_obs_dis = self._trajectory_manager.reference_spline_last_obs_dis - \
                                      self._trajectory_manager.reference_spline_current_dis

            action_spline_length = self._trajectory_manager.action_spline.get_length()

            if action_spline_length >= current_to_last_obs_dis:
                if self._spline_final_distance_reward is None or not self._trajectory_manager.spline_finished:
                    # decreasing reward, as the action spline is longer as desired
                    rel_dis = min(1.0, (action_spline_length - current_to_last_obs_dis) /
                                  (self._spline_final_overshoot_factor * self._obs_spline_n_next *
                                   self._trajectory_manager.spline_max_distance_between_knots))
                    # rel_dis in [0, 1] -> the distance_reward is 1 if rel_dis is 0 and decreases
                    # to zero for rel_dis == 1
                    # the distance is computed relative to n_next * max_distance_between_knots
                    # (Note: it would also be possible to use a different distance for normalization.
                    # The intention here is that the length of the action spline usually does not exceed this distance)
                    distance_reward = (rel_dis - 1) ** 2
                    # quadratic decrease from 1 to 0, highest decrease for rel_dis == 0
                else:
                    distance_reward = self._spline_final_distance_reward
            else:
                # increasing reward as the action spline is shorter as desired
                rel_dis = action_spline_length / current_to_last_obs_dis  # in [0, 1)
                distance_reward = - (rel_dis - 1) ** 2 + 1
                # quadratic increase from 0 to 1, highest increase for rel_dis == 0

            if self._spline_speed_range is not None:
                distance_reward_speed = distance_reward * self._spline_speed
            else:
                distance_reward_speed = distance_reward

            if self._punish_action:
                action_punishment = self._compute_action_punishment()  # action punishment

            if self._reward_action:
                if self._trajectory_manager.spline_close_to_end:
                    action_reward = 1.0
                else:
                    action_reward = self._compute_action_reward()

            if self._punish_spline_max_deviation:
                rel_deviation = min(1.0, self._trajectory_manager.spline_max_deviation /
                                    self._spline_deviation_max_threshold)
                max_deviation_punishment = - (rel_deviation - 1) ** 2 + 1
                # quadratic increase from 0 to 1, highest increase for rel_deviation == 0

            if self._punish_spline_mean_deviation:
                rel_deviation = min(1.0, self._trajectory_manager.spline_mean_deviation /
                                    self._spline_deviation_max_threshold)
                mean_deviation_punishment = - (rel_deviation - 1) ** 2 + 1
                # quadratic increase from 0 to 1, highest increase for rel_deviation == 0

            if self._punish_spline_max_cartesian_deviation:
                rel_deviation = min(1.0, self._trajectory_manager.spline_max_cartesian_deviation /
                                    self._spline_cartesian_deviation_max_threshold)
                max_cartesian_deviation_punishment = - (rel_deviation - 1) ** 2 + 1
                # quadratic increase from 0 to 1, highest increase for rel_deviation == 0

            if self._punish_spline_mean_cartesian_deviation:
                rel_deviation = min(1.0, self._trajectory_manager.spline_mean_cartesian_deviation /
                                    self._spline_deviation_max_threshold)
                mean_cartesian_deviation_punishment = - (rel_deviation - 1) ** 2 + 1
                # quadratic increase from 0 to 1, highest increase for rel_deviation == 0

            if self._spline_speed_range is not None:
                spline_max_deviation_max_punishment = self._spline_max_deviation_max_punishment / self._spline_speed
                spline_mean_deviation_max_punishment = self._spline_mean_deviation_max_punishment / self._spline_speed
            else:
                spline_max_deviation_max_punishment = self._spline_max_deviation_max_punishment
                spline_mean_deviation_max_punishment = self._spline_mean_deviation_max_punishment

            if self._robot_scene.sphere_balancing_mode:
                balancing_sphere_reward_list = \
                    [self._compute_sphere_balancing_reward(robot=i) for i in range(self._robot_scene.num_robots)]
                balancing_sphere_reward = np.mean(balancing_sphere_reward_list)

            if self._robot_scene.robot_base_balancing_mode:
                # pos deviation
                rel_deviation = min(1.0, self._robot_scene.robot_base_pos_deviation /
                                    self._balancing_robot_base_max_pos_deviation)
                balancing_robot_base_pos_reward = (rel_deviation - 1) ** 2
                # quadratic decrease from 1 to 0, highest decrease for rel_deviation == 0
                # orn deviation
                rel_deviation = min(1.0, self._robot_scene.robot_base_orn_deviation /
                                    self._balancing_robot_base_max_orn_deviation)
                balancing_robot_base_orn_reward = (rel_deviation - 1) ** 2
                # quadratic decrease from 1 to 0, highest decrease for rel_deviation == 0
                # z angle deviation
                rel_deviation = min(1.0, self._robot_scene.robot_z_angle_deviation_rad /
                                    self._balancing_robot_base_max_z_angle_deviation_rad)
                balancing_robot_base_z_angle_reward = (rel_deviation - 1) ** 2

                if self._balancing_robot_base_punish_last_cartesian_action_point:
                    # punish cartesian deviation between the floating last cartesian action point and the static
                    # target point
                    last_cartesian_action_point = self._trajectory_manager.spline_last_cartesian_action_point
                    cartesian_deviation_list = []
                    for i in range(len(last_cartesian_action_point)):
                        floating_action_point_pos, floating_action_point_orn = \
                            self._robot_scene.convert_point_from_static_base_to_floating_base(
                                pos_static=last_cartesian_action_point[i][:3],
                                orn_static=last_cartesian_action_point[i][3:])
                        cartesian_deviation_list.append(
                            np.linalg.norm(floating_action_point_pos -
                                           self._trajectory_manager.spline_target_point[i][:3]))
                    cartesian_deviation = np.mean(cartesian_deviation_list)
                    rel_deviation = \
                        min(1.0, cartesian_deviation /
                            self._balancing_robot_base_spline_last_cartesian_action_point_deviation_max_threshold)
                    balancing_robot_base_last_cartesian_action_point_punishment = - (rel_deviation - 1) ** 2 + 1
                    # quadratic increase from 0 to 1, highest increase for rel_deviation == 0

            distance_deviation_reward = distance_reward_speed * self._spline_distance_max_reward \
                + (1 - max_deviation_punishment) * spline_max_deviation_max_punishment \
                + (1 - mean_deviation_punishment) * spline_mean_deviation_max_punishment

            reward = distance_deviation_reward \
                + (1 - action_punishment) * self._action_max_punishment \
                + action_reward * self._action_max_reward \
                + (1 - max_cartesian_deviation_punishment) * self._spline_max_cartesian_deviation_max_punishment \
                + (1 - mean_cartesian_deviation_punishment) * self._spline_mean_cartesian_deviation_max_punishment \
                + balancing_sphere_reward * self._balancing_sphere_max_reward \
                + balancing_robot_base_pos_reward * self._balancing_robot_base_pos_max_reward \
                + balancing_robot_base_orn_reward * self._balancing_robot_base_orn_max_reward \
                + balancing_robot_base_z_angle_reward * self._balancing_robot_base_z_angle_max_reward \
                + (1 - balancing_robot_base_last_cartesian_action_point_punishment) * \
                self._balancing_robot_base_spline_last_cartesian_action_point_deviation_max_punishment

        for key in ['average', 'min', 'max']:
            info[key].update(action_punishment=action_punishment,
                             action_reward=action_reward,
                             distance_reward=distance_reward,
                             distance_reward_speed=distance_reward_speed,
                             distance_deviation_reward=distance_deviation_reward,
                             max_deviation_punishment=max_deviation_punishment,
                             mean_deviation_punishment=mean_deviation_punishment,
                             mean_deviation=self._trajectory_manager.spline_mean_deviation,
                             max_deviation=self._trajectory_manager.spline_max_deviation,
                             mean_cartesian_deviation_punishment=mean_cartesian_deviation_punishment,
                             max_cartesian_deviation_punishment=max_cartesian_deviation_punishment,
                             mean_cartesian_deviation=self._trajectory_manager.spline_mean_cartesian_deviation,
                             max_cartesian_deviation=self._trajectory_manager.spline_max_cartesian_deviation,
                             )

        info['sum'].update(distance_reward_speed=distance_reward_speed,
                           distance_deviation_reward=distance_deviation_reward)

        if self._robot_scene.sphere_balancing_mode:
            for key in ['average', 'min', 'max', 'sum']:
                info[key].update(balancing_sphere_reward=balancing_sphere_reward)

        if self._robot_scene.robot_base_balancing_mode:
            for key in ['average', 'min', 'max', 'sum']:
                info[key].update(balancing_robot_base_pos_reward=balancing_robot_base_pos_reward,
                                 balancing_robot_base_orn_reward=balancing_robot_base_orn_reward,
                                 balancing_robot_base_z_angle_reward=balancing_robot_base_z_angle_reward,
                                 balancing_robot_base_last_cartesian_action_point_punishment=
                                 balancing_robot_base_last_cartesian_action_point_punishment,
                                 balancing_robot_base_z_angle_deg=
                                 self._robot_scene.robot_z_angle_deviation_rad * 180 / np.pi)

        reward, info = self._common_reward_function(reward=reward, info=info)

        return reward, info

    def _compute_sphere_balancing_reward(self, robot=0):
        sphere_pos_local, sphere_pos_local_rel = self._robot_scene.get_sphere_position_on_board(robot=robot)

        if self._robot_scene.is_sphere_on_board(sphere_pos_local_rel):
            # sphere on board
            if self._balancing_sphere_dev_min_max is None:
                balancing_rewards = (1 - np.array(sphere_pos_local_rel)) / \
                                    (1 - self._robot_scene.BOARD_SPHERE_POSITIONING_OFFSET)
                balancing_reward = min(balancing_rewards[:2])

                return max(min(balancing_reward, 1), 0) ** 2
            else:
                sphere_distance = np.linalg.norm(
                    np.array(sphere_pos_local) - np.array(self._robot_scene.sphere_start_pos_list[robot]))
                balancing_reward = (self._balancing_sphere_dev_min_max[0] - sphere_distance) \
                    / (self._balancing_sphere_dev_min_max[1] - self._balancing_sphere_dev_min_max[0]) + 1

                return max(min(balancing_reward, 1), 0) ** 2
        else:
            return 0
