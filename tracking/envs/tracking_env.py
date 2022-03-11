# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import inspect
import numpy as np
import logging


from tracking.envs.decorators import actions, observations, rewards, video_recording
from tracking.utils.spline_plotter import SplinePlotter

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


class TrackingEnv(actions.AccelerationPredictionBoundedJerkAccVelPos,
                  observations.TrackingObservation,
                  rewards.TargetPointReachingReward,
                  video_recording.VideoRecordingManager):
    def __init__(self,
                 *vargs,
                 **kwargs):
        super().__init__(*vargs, **kwargs)


class TrackingEnvSpline(actions.AccelerationPredictionBoundedJerkAccVelPos,
                        observations.TrackingObservation,
                        rewards.SplineTrackingReward,
                        video_recording.VideoRecordingManager):
    def __init__(self,
                 spline_max_final_deviation=0.05,
                 spline_braking_extra_time_steps=None,
                 *vargs,
                 **kwargs):
        self._spline_max_final_deviation = spline_max_final_deviation
        self._spline_braking_extra_time_steps = spline_braking_extra_time_steps
        self._spline_finished_time = None
        self._spline_finished_time_braking = None
        self._spline_close_to_end_time = None
        self._action_spline_total_length = None

        super().__init__(*vargs, **kwargs)
        if self._plot_spline:
            self._spline_plotter = SplinePlotter(plot_norm=True, normalize=True,
                                                 pos_limits=self.pos_limits_min_max.T)

    def reset(self, **kwargs):
        self._spline_finished_time = 0  # the time at which the end of the spline is reached
        self._spline_finished_time_braking = 0  # the time at which the robot is stopped after the end of the spline
        # has been reached
        self._spline_close_to_end_time = 0  # the time at which the end of the spline is part of the observation
        # zero to avoid problems with the custom metrics when the spline is terminated due to the maximum deviation
        # for evaluation, only those episodes that reached the end of the spline should be considered

        self._action_spline_total_length = 0
        if self._spline_speed_range is not None and not self._spline_speed_fixed:
            self._spline_speed = np.random.uniform(self._spline_speed_range[0],
                                                   self._spline_speed_range[1])
            if not self._spline_random_speed_per_time_step:
                logging.info("Spline speed this episode: %s", self._spline_speed)
        return super().reset(**kwargs)

    def _prepare_for_next_action(self):
        super()._prepare_for_next_action()

    def _process_action_outcome(self, base_info, action_info, robot_stopped=False):
        # length of the action spline compared to the length of the observation spline

        action_spline_length = self._trajectory_manager.action_spline.get_length()
        self._action_spline_total_length = self._action_spline_total_length + action_spline_length
        if self._trajectory_manager.reference_spline_last_obs_dis > \
                self._trajectory_manager.reference_spline_current_dis:
            action_spline_observation_fraction = action_spline_length / \
                (self._trajectory_manager.reference_spline_last_obs_dis -
                 self._trajectory_manager.reference_spline_current_dis)
        else:
            action_spline_observation_fraction = 1.0

        if self._use_splines:
            spline_finished = self._trajectory_manager.update_current_spline_dis()
            if spline_finished and self._spline_finished_time == 0:
                self._spline_finished_time = self._episode_length * self._trajectory_time_step
            if spline_finished and self._spline_braking_extra_time_steps is not None:
                if self._trajectory_manager.spline_extra_time_steps >= self._spline_braking_extra_time_steps:
                    self._brake = True
            if spline_finished and robot_stopped and self._spline_finished_time_braking == 0:
                self._spline_finished_time_braking = self._episode_length * self._trajectory_time_step

        for key in ['average', 'min', 'max']:
            base_info[key]['action_spline_observation_fraction'] = action_spline_observation_fraction
            base_info[key]['action_spline_length'] = action_spline_length
            if not self._trajectory_manager.spline_close_to_end:
                base_info[key]['action_spline_length_not_at_spline_end'] = action_spline_length
            else:
                if self._spline_close_to_end_time == 0:
                    self._spline_close_to_end_time = self._episode_length * self._trajectory_time_step

        return super()._process_action_outcome(base_info, action_info, robot_stopped)

    def _process_end_of_episode(self, observation, reward, done, info):
        spline_final_deviation = self._trajectory_manager.get_final_spline_deviation()
        if self._trajectory_successful:
            # trajectory_successful is set to True at the beginning of each episode
            # check if the trajectory was really successful
            if (self._termination_reason == self.TERMINATION_SPLINE_LENGTH or
                self._termination_reason == self.TERMINATION_TRAJECTORY_LENGTH or
                self._termination_reason == self.TERMINATION_ROBOT_STOPPED) and \
                    spline_final_deviation < self._spline_max_final_deviation:
                # might make sense to add a constraint on the final velocity here
                self._trajectory_successful = True
            else:
                self._trajectory_successful = False
        # compute trajectory fraction ->
        # (current_spline_dis - initial_spline_dis) / (spline_length - initial_spline_dis)
        reference_spline_length = self._trajectory_manager.reference_spline.end_dis \
            - self._trajectory_manager.reference_spline.start_dis

        info['trajectory_fraction'] = (self._trajectory_manager.reference_spline_current_dis -
                                       self._trajectory_manager.reference_spline.start_dis) / reference_spline_length

        info['reference_spline_length'] = reference_spline_length
        info['action_spline_total_length'] = self._action_spline_total_length
        info['spline_final_deviation'] = spline_final_deviation
        info['spline_finished_time'] = self._spline_finished_time if self._spline_finished_time != 0.0 else np.nan
        info['spline_close_to_end_time'] = self._spline_close_to_end_time \
            if self._spline_close_to_end_time != 0.0 else np.nan
        info['spline_max_end_deviation'] = self._trajectory_manager.spline_max_end_deviation \
            if self._trajectory_manager.spline_max_end_deviation != 0.0 else np.nan
        info['spline_first_end_deviation'] = self._trajectory_manager.spline_first_end_deviation \
            if self._trajectory_manager.spline_first_end_deviation != 0.0 else np.nan
        info['spline_finished_time_braking'] = self._spline_finished_time_braking \
            if self._spline_finished_time_braking != 0.0 else np.nan

        if self._spline_finished_time == 0:
            info['spline_finished_rate'] = 0.0
        else:
            info['spline_finished_rate'] = 1.0

        if self._spline_close_to_end_time == 0:
            info['spline_close_to_end_rate'] = 0.0
        else:
            info['spline_close_to_end_rate'] = 1.0

        if self._plot_spline:
            self._spline_plotter.reset_plotter()
            self._spline_plotter.add_joint_spline(spline=self._trajectory_manager.reference_spline,
                                                  linestyle='-',
                                                  label='Reference', marker='o')
            self._spline_plotter.add_joint_spline(spline=self._trajectory_manager.total_spline,
                                                  linestyle=':',
                                                  label='_nolegend_', marker=None)
            self._spline_plotter.display_plot()

        if self._spline_compute_total_spline_metrics:
            total_spline_mean_deviation, total_spline_max_deviation, total_spline_final_deviation, \
                total_spline_mean_cartesian_pos_deviation, total_spline_max_cartesian_pos_deviation, \
                total_spline_final_cartesian_pos_deviation, total_spline_mean_cartesian_orn_deviation_rad, \
                total_spline_max_cartesian_orn_deviation_rad, total_spline_final_cartesian_orn_deviation_rad \
                = self._trajectory_manager.get_total_spline_deviation()
            info['spline_total_spline_mean_deviation'] = total_spline_mean_deviation
            info['spline_total_spline_max_deviation'] = total_spline_max_deviation
            info['spline_total_spline_final_deviation'] = total_spline_final_deviation
            info['spline_total_spline_mean_cartesian_pos_deviation'] = total_spline_mean_cartesian_pos_deviation
            info['spline_total_spline_max_cartesian_pos_deviation'] = total_spline_max_cartesian_pos_deviation
            info['spline_total_spline_final_cartesian_pos_deviation'] = total_spline_final_cartesian_pos_deviation
            info['spline_total_spline_mean_cartesian_orn_deviation_deg'] = \
                total_spline_mean_cartesian_orn_deviation_rad * 180 / np.pi
            info['spline_total_spline_max_cartesian_orn_deviation_deg'] = \
                total_spline_max_cartesian_orn_deviation_rad * 180 / np.pi
            info['spline_total_spline_final_cartesian_orn_deviation_deg'] = \
                total_spline_final_cartesian_orn_deviation_rad * 180 / np.pi

        if self._robot_scene.floating_robot_base and logging.root.level <= logging.INFO:
            robot_z_angle_deviation_rad_max = np.max(info['max']['robot_z_angle_deviation_rad'])
            logging.info("Max z angle: %s deg", robot_z_angle_deviation_rad_max / np.pi * 180)

        return super()._process_end_of_episode(observation, reward, done, info)

