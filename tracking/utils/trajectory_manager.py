# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os.path
import json
import numpy as np
import glob
import logging


from tracking.utils.spline_utils import Spline, convert_joint_space_to_cartesian_space

TERMINATION_TRAJECTORY_LENGTH = 2
TERMINATION_SPLINE_LENGTH = 3
TERMINATION_SPLINE_DEVIATION = 4


def clip_index(index, list_length):
    if index < 0 and abs(index) > list_length:
        return 0
    if index > 0 and index > list_length - 1:
        return -1
    else:
        return index


class TrajectoryManager(object):

    def __init__(self,
                 trajectory_time_step,
                 trajectory_duration,
                 obstacle_wrapper,
                 use_splines=False,
                 spline_u_arc_start_range=(0, 0),
                 spline_u_arc_diff_min=1.0,
                 spline_u_arc_diff_max=1.0,
                 spline_dir=None,
                 spline_config_path=None,
                 visualize_action_spline=False,
                 spline_termination_max_deviation=0.25,
                 spline_termination_extra_time_steps=None,
                 spline_deviation_sample_distance=0.01,
                 spline_deviation_weighting_factors=None,
                 spline_normalize_duration=False,
                 spline_compute_cartesian_deviation=False,
                 spline_use_reflection_vectors=False,
                 env=None,
                 **kwargs):

        self._trajectory_time_step = trajectory_time_step
        self._trajectory_duration = trajectory_duration
        self._num_time_steps = int(trajectory_duration / trajectory_time_step)
        self._obstacle_wrapper = obstacle_wrapper
        self._use_splines = use_splines
        self._spline_dir = spline_dir
        self._spline_config_path = spline_config_path
        self._spline_u_arc_start_range = spline_u_arc_start_range
        self._spline_u_arc_diff_min = spline_u_arc_diff_min
        self._spline_u_arc_diff_max = spline_u_arc_diff_max
        self._spline_max_distance_between_knots = None
        self._spline_use_normalized_length_correction_step_size = None
        self._spline_length_correction_step_size = None
        self._spline_reflection_vectors = None
        self._reference_spline_data = {}
        self._spline_name_list = []
        if self._use_splines and self._spline_dir is not None:
            self._load_spline_config()
            self._spline_name_list = sorted(glob.glob(os.path.join(self._spline_dir, "episode_*.json")))
            self._spline_name_list = [os.path.basename(path) for path in self._spline_name_list]

        self._trajectory_start_position = None
        self._trajectory_length = None
        self._num_manip_joints = None
        self._zero_joint_vector = None
        self._generated_trajectory = None
        self._measured_actual_trajectory_control_points = None
        self._computed_actual_trajectory_control_points = None
        self._generated_trajectory_control_points = None

        self._controller_model_coefficient_a = None
        self._controller_model_coefficient_b = None

        self._reference_spline = None
        self._spline_deviation_sample_distance = spline_deviation_sample_distance
        self._spline_termination_max_deviation = spline_termination_max_deviation
        self._reference_spline_current_dis = None
        self._reference_spline_last_obs_dis = None
        self._spline_name = None
        self._spline_max_deviation = None
        self._spline_mean_deviation = None
        self._spline_first_end_deviation = None
        self._spline_max_end_deviation = None
        self._spline_max_cartesian_deviation = None
        self._spline_mean_cartesian_deviation = None
        self._spline_target_point = None
        self._spline_target_point_relative_pos = None
        self._spline_last_cartesian_action_point = None
        self._spline_compute_cartesian_deviation = spline_compute_cartesian_deviation
        self._spline_use_reflection_vectors = spline_use_reflection_vectors
        self._spline_termination_extra_time_steps = spline_termination_extra_time_steps
        self._spline_extra_time_steps = None  # time steps after the end of the reference_spline is reached
        self._spline_normalize_duration = spline_normalize_duration
        # normalize the trajectory duration based on the fraction of the spline that is actually used
        self._spline_deviation_weighting_factors = spline_deviation_weighting_factors
        if self._spline_deviation_weighting_factors is not None:
            self._spline_deviation_weighting_factors = np.asarray(self._spline_deviation_weighting_factors)
            self._spline_deviation_normalizing_factor = \
                np.sqrt(len(self._spline_deviation_weighting_factors) /
                        np.sum(np.square(self._spline_deviation_weighting_factors)))

        self._action_spline = None
        self._total_spline = None
        self._visualize_action_spline = visualize_action_spline
        self._action_spline_debug_line_buffer = None

        self._env = env

        if self._env._use_gui and self._use_splines:
            if self._visualize_action_spline:
                logging.info("Preparing action spline visualization. This might take some time ...")
            else:
                logging.info("Action spline visualization deactivated. Activate with --visualize_action_spline")

    @property
    def generated_trajectory_control_points(self):
        return self._generated_trajectory_control_points

    @property
    def measured_actual_trajectory_control_points(self):
        return self._measured_actual_trajectory_control_points

    @property
    def computed_actual_trajectory_control_points(self):
        return self._computed_actual_trajectory_control_points

    @property
    def trajectory_time_step(self):
        return self._trajectory_time_step

    @property
    def trajectory_length(self):
        return self._trajectory_length

    @property
    def reference_spline(self):
        return self._reference_spline

    @property
    def action_spline(self):
        return self._action_spline

    @property
    def spline_max_distance_between_knots(self):
        return self._spline_max_distance_between_knots

    @spline_max_distance_between_knots.setter
    def spline_max_distance_between_knots(self, val):
        self._spline_max_distance_between_knots = val

    @property
    def spline_use_normalized_length_correction_step_size(self):
        return self._spline_use_normalized_length_correction_step_size

    @spline_use_normalized_length_correction_step_size.setter
    def spline_use_normalized_length_correction_step_size(self, val):
        self._spline_use_normalized_length_correction_step_size = val

    @property
    def spline_length_correction_step_size(self):
        return self._spline_length_correction_step_size

    @spline_length_correction_step_size.setter
    def spline_length_correction_step_size(self, val):
        self._spline_length_correction_step_size = val

    @property
    def reference_spline_current_dis(self):
        return self._reference_spline_current_dis

    @property
    def reference_spline_last_obs_dis(self):
        return self._reference_spline_last_obs_dis

    @reference_spline_last_obs_dis.setter
    def reference_spline_last_obs_dis(self, val):
        self._reference_spline_last_obs_dis = val

    @property
    def spline_max_deviation(self):
        if self._spline_max_deviation is None:
            self._compute_spline_deviation()
        return self._spline_max_deviation

    @property
    def spline_mean_deviation(self):
        if self._spline_mean_deviation is None:
            self._compute_spline_deviation()
        return self._spline_mean_deviation

    @property
    def spline_max_end_deviation(self):
        if self._spline_max_end_deviation is None:  # spline not at end
            return 0.0
        return self._spline_max_end_deviation

    @property
    def spline_first_end_deviation(self):
        if self._spline_first_end_deviation is None:  # spline not at end
            return 0.0
        return self._spline_first_end_deviation

    @property
    def spline_max_cartesian_deviation(self):
        if self._spline_compute_cartesian_deviation:
            if self._spline_max_cartesian_deviation is None:
                self._compute_spline_deviation()
            return self._spline_max_cartesian_deviation
        return 0.0  # cartesian deviation not required

    @property
    def spline_mean_cartesian_deviation(self):
        if self._spline_compute_cartesian_deviation:
            if self._spline_mean_cartesian_deviation is None:
                self._compute_spline_deviation()
            return self._spline_mean_cartesian_deviation
        else:
            return 0.0  # cartesian deviation not required

    @property
    def spline_last_cartesian_action_point(self):
        if self._spline_compute_cartesian_deviation:
            if self._spline_last_cartesian_action_point is None:
                self._compute_spline_deviation()
            return self._spline_last_cartesian_action_point
        else:
            return 0.0  # cartesian deviation not required

    @property
    def total_spline(self):
        if self._total_spline is None:
            self._total_spline = self._compute_total_spline()
        return self._total_spline

    @property
    def spline_target_point(self):
        return self._spline_target_point

    @property
    def spline_target_point_relative_pos(self):
        return self._spline_target_point_relative_pos

    @property
    def spline_close_to_end(self):
        return self.reference_spline_last_obs_dis == self.reference_spline.end_dis

    @property
    def spline_finished(self):
        return self._spline_extra_time_steps >= 0

    @property
    def spline_name(self):
        return self._spline_name

    @property
    def spline_extra_time_steps(self):
        return self._spline_extra_time_steps

    def reset(self, get_new_trajectory=True, spline_name=None, duration_multiplier=None):
        if get_new_trajectory:
            if self._use_splines:
                if spline_name is None:
                    random_spline_index = np.random.randint(low=0, high=len(self._spline_name_list))
                    spline_name = self._spline_name_list[random_spline_index]

                self._spline_name = spline_name
                self._reference_spline = self.get_reference_spline(spline_name=spline_name)
                u_arc_start = np.random.uniform(self._spline_u_arc_start_range[0], self._spline_u_arc_start_range[1])
                u_arc_end = np.random.uniform(min(u_arc_start + self._spline_u_arc_diff_min, 1.0),
                                              min(u_arc_start + self._spline_u_arc_diff_max, 1.0))
                self._reference_spline.set_u_start(u_arc_start=u_arc_start)
                self._reference_spline.set_u_end_index(u_arc_end_min=u_arc_end)
                logging.info("Using spline %s (u arc start %s, u arc end %s).", spline_name, u_arc_start, u_arc_end)
                reference_spline_start_position = self._reference_spline.get_joint_position(u_arc=u_arc_start)
                self._trajectory_start_position = np.zeros(self._env._robot_scene.num_manip_joints)
                self._trajectory_start_position[self._env._robot_scene.spline_joint_mask] = \
                    reference_spline_start_position
                if self._spline_compute_cartesian_deviation:
                    self._spline_target_point = \
                        convert_joint_space_to_cartesian_space(self._env, reference_spline_start_position)[:, 0, :]
                    self._spline_target_point_relative_pos = np.zeros_like(self._spline_target_point)
                if self._spline_normalize_duration:
                    trajectory_duration = self._trajectory_duration * \
                                          (self._reference_spline.end_dis - self._reference_spline.start_dis) / \
                                          self._reference_spline.get_length()
                else:
                    trajectory_duration = self._trajectory_duration

            else:
                self._trajectory_start_position = self._get_new_trajectory_start_position()
                trajectory_duration = self._trajectory_duration

            if duration_multiplier is not None:
                trajectory_duration = trajectory_duration * duration_multiplier

            self._trajectory_length = round(trajectory_duration / self._trajectory_time_step) + 1
        self._num_manip_joints = len(self._trajectory_start_position)
        self._zero_joint_vector = np.array([0.0] * self._num_manip_joints)
        self._generated_trajectory = {'positions': [self.get_trajectory_start_position()],
                                      'velocities': [self._zero_joint_vector],
                                      'accelerations': [self._zero_joint_vector]}
        self._measured_actual_trajectory_control_points = {'positions': [self.get_trajectory_start_position()],
                                                           'velocities': [self._zero_joint_vector],
                                                           'accelerations': [self._zero_joint_vector]}
        self._computed_actual_trajectory_control_points = {'positions': [self.get_trajectory_start_position()],
                                                           'velocities': [self._zero_joint_vector],
                                                           'accelerations': [self._zero_joint_vector]}
        self._generated_trajectory_control_points = {'positions': [self.get_trajectory_start_position()],
                                                     'velocities': [self._zero_joint_vector],
                                                     'accelerations': [self._zero_joint_vector]}

        self._action_spline = None
        self._total_spline = None
        self._spline_max_deviation = None
        self._spline_mean_deviation = None
        self._spline_max_cartesian_deviation = None
        self._spline_mean_cartesian_deviation = None
        self._spline_last_cartesian_action_point = None
        self._spline_first_end_deviation = None
        self._spline_max_end_deviation = None
        self._spline_extra_time_steps = -1
        self._reference_spline_current_dis = None
        self._reference_spline_last_obs_dis = None

        if self._env._use_gui and self._visualize_action_spline:
            # reset action spline for video rendering with imagegrab
            self._action_spline_debug_line_buffer = \
                Spline.reset_debug_line_buffer(self._action_spline_debug_line_buffer,
                                               physics_client_id=self._env._gui_client_id)

    def _load_spline_config(self):
        spline_config_path = os.path.join(os.path.dirname(self._spline_dir), "spline_config.json") \
            if self._spline_config_path is None else self._spline_config_path
        if os.path.exists(spline_config_path):
            with open(spline_config_path) as file:
                spline_config_dict = json.load(file)
            self._spline_max_distance_between_knots = spline_config_dict["max_dis_between_knots"]
            self._spline_use_normalized_length_correction_step_size = \
                spline_config_dict["use_normalized_length_correction_step_size"]
            self._spline_length_correction_step_size = spline_config_dict["length_correction_step_size"]
            if "reflection_vectors" in spline_config_dict:
                self._spline_reflection_vectors = spline_config_dict["reflection_vectors"]

        else:
            raise ValueError("Could not find reference_spline config {}".format(spline_config_path))

    def add_reference_spline(self, spline_name, spline_data):
        self._reference_spline_data[spline_name] = spline_data

    def get_reference_spline(self, spline_name):
        if spline_name not in self._reference_spline_data:
            # load reference_spline from dataset
            spline_path = os.path.join(self._spline_dir, spline_name)
            spline_reflection_vectors = self._spline_reflection_vectors if self._spline_use_reflection_vectors else None
            if os.path.exists(spline_path):
                self._reference_spline_data[spline_name] = \
                    Spline.load_from_json(spline_path,
                                          length_correction_step_size=self._spline_length_correction_step_size,
                                          use_normalized_length_correction_step_size=
                                          self._spline_use_normalized_length_correction_step_size,
                                          reflection_vectors=spline_reflection_vectors)
            else:
                raise FileNotFoundError("Could not find reference_spline {}".format(spline_path))

        self._reference_spline_data[spline_name].reset(random_reflection_vector_index=True)
        return self._reference_spline_data[spline_name]

    def _compute_total_spline(self):
        # returns a spline object generated from all generated trajectory control points
        curve_data = np.array(
            self._generated_trajectory_control_points['positions'])[:, self._env._robot_scene.spline_joint_mask].T
        total_spline = Spline(curve_data=curve_data,
                              curve_data_slicing_step=1,
                              length_correction_step_size=self._spline_length_correction_step_size,
                              use_normalized_length_correction_step_size=
                              self._spline_use_normalized_length_correction_step_size,
                              method="auto",
                              curvature_at_ends=0)
        return total_spline

    def generate_action_spline(self, knot_data):
        self._spline_max_deviation = None
        # reset deviation between the action reference_spline and the reference reference_spline
        self._spline_mean_deviation = None
        self._spline_max_cartesian_deviation = None
        self._spline_mean_cartesian_deviation = None
        self._spline_last_cartesian_action_point = None

        self._reference_spline_current_dis = self._reference_spline.current_dis  # copy the current dis of the time step

        append_spline = False if self._action_spline is None else True
        knots_per_time_step = knot_data.shape[1]

        self._action_spline = Spline(curve_data=knot_data, curve_data_slicing_step=1,
                                     length_correction_step_size=self._spline_length_correction_step_size,
                                     use_normalized_length_correction_step_size=
                                     self._spline_use_normalized_length_correction_step_size,
                                     method="auto",
                                     curvature_at_ends=0)

        if self._env._use_gui and self._visualize_action_spline:
            self._action_spline_debug_line_buffer = \
                self._action_spline.visualize(env=self._env,
                                              debug_line_buffer=self._action_spline_debug_line_buffer,
                                              append_spline=append_spline,
                                              visualize_knots=False,
                                              visualize_knots_orn=False,
                                              debug_line_buffer_init_size=
                                              self._num_time_steps * (knots_per_time_step + 2),
                                              line_color=[1, 0, 0],
                                              line_width=2,
                                              u_marker=[0.0],
                                              marker_color=[91/255, 20/255, 15/255],
                                              marker_width=3,
                                              physics_client_id=self._env._gui_client_id)

    def update_current_spline_dis(self):
        spline_finished = self._reference_spline.add_dis_to_current_dis(self._action_spline.get_length())
        if spline_finished:
            self._spline_extra_time_steps += 1
        return spline_finished

    def _compute_spline_deviation(self):
        action_spline_sampling_interpolation = self._action_spline.get_interpolation(
            sample_distance=self._spline_deviation_sample_distance,
            use_normalized_sample_distance=False, convert_to_u=False, lin_space=True)[1:]

        reference_spline_sampling_interpolation = \
            action_spline_sampling_interpolation + self._reference_spline_current_dis
        sampling_indices_at_reference_spline_end = \
            reference_spline_sampling_interpolation >= self._reference_spline.end_dis
        sampling_indices_not_at_reference_spline_end = np.invert(sampling_indices_at_reference_spline_end)
        reference_spline_sampling_interpolation[sampling_indices_at_reference_spline_end] = \
            self._reference_spline.end_dis

        action_spline_sampling_points = self._action_spline.evaluate_length(action_spline_sampling_interpolation).T
        reference_spline_sampling_points = np.zeros_like(action_spline_sampling_points)

        if np.any(sampling_indices_at_reference_spline_end):
            reference_spline_sampling_points[sampling_indices_at_reference_spline_end, :] = \
                self._reference_spline.curve_data_spline[:, self._reference_spline.u_end_index]

        if np.any(sampling_indices_not_at_reference_spline_end):
            reference_spline_sampling_points[sampling_indices_not_at_reference_spline_end, :] = \
                self._reference_spline.evaluate_length(
                    reference_spline_sampling_interpolation[sampling_indices_not_at_reference_spline_end]).T

        if self._spline_deviation_weighting_factors is not None:
            weighted_deviation = (reference_spline_sampling_points - action_spline_sampling_points) \
                                 * self._spline_deviation_weighting_factors
            spline_deviation = self._spline_deviation_normalizing_factor * np.linalg.norm(weighted_deviation, axis=1)
        else:
            spline_deviation = np.linalg.norm(reference_spline_sampling_points - action_spline_sampling_points, axis=1)

        if np.any(sampling_indices_at_reference_spline_end):
            max_end_deviation = np.max(spline_deviation[sampling_indices_at_reference_spline_end])
            if self._spline_max_end_deviation is None:
                self._spline_max_end_deviation = max_end_deviation
            else:
                self._spline_max_end_deviation = max(self._spline_max_end_deviation, max_end_deviation)

            if np.any(sampling_indices_not_at_reference_spline_end):
                # time step at which the end of the spline is reached
                action_spline_first_end_length = self._reference_spline.end_dis - self._reference_spline_current_dis
                action_spline_first_end_point = self._action_spline.evaluate_length(action_spline_first_end_length)

                if self._spline_deviation_weighting_factors is not None:
                    weighted_deviation = (self._reference_spline.curve_data_spline[:,
                                          self._reference_spline.u_end_index] -
                                          action_spline_first_end_point) \
                                         * self._spline_deviation_weighting_factors
                    self._spline_first_end_deviation = self._spline_deviation_normalizing_factor * \
                        np.linalg.norm(weighted_deviation, axis=0)
                else:
                    self._spline_first_end_deviation = np.linalg.norm(self._reference_spline.curve_data_spline[:,
                                                                      self._reference_spline.u_end_index] -
                                                                      action_spline_first_end_point, axis=0)

        self._spline_max_deviation = np.max(spline_deviation)
        self._spline_mean_deviation = np.mean(spline_deviation)

        if self._spline_compute_cartesian_deviation:
            action_spline_cartesian_sampling_points = \
                convert_joint_space_to_cartesian_space(self._env, joint_data=action_spline_sampling_points.T)
            reference_spline_cartesian_sampling_points = np.zeros_like(action_spline_cartesian_sampling_points)
            if np.any(sampling_indices_at_reference_spline_end):
                reference_spline_cartesian_sampling_points[:, sampling_indices_at_reference_spline_end, :] = \
                    convert_joint_space_to_cartesian_space(
                        self._env,
                        joint_data=self._reference_spline.curve_data_spline[:, self._reference_spline.u_end_index])
            if np.any(sampling_indices_not_at_reference_spline_end):
                reference_spline_cartesian_sampling_points[:, sampling_indices_not_at_reference_spline_end, :] = \
                    convert_joint_space_to_cartesian_space(
                        self._env,
                        joint_data=reference_spline_sampling_points[sampling_indices_not_at_reference_spline_end, :].T)

            spline_cartesian_deviation = np.linalg.norm(action_spline_cartesian_sampling_points[:, :, 0:3] -
                                                        reference_spline_cartesian_sampling_points[:, :, 0:3], axis=2)

            self._spline_max_cartesian_deviation = np.max(spline_cartesian_deviation)
            self._spline_mean_cartesian_deviation = np.mean(spline_cartesian_deviation)

            # cartesian pos (and orn) of the current point of the reference spline
            self._spline_target_point = reference_spline_cartesian_sampling_points[:, -1, :]
            # cartesian relative pos (and orn) of the current point of the reference spline relative
            # to the current pos of the target link
            self._spline_target_point_relative_pos = self._spline_target_point - \
                action_spline_cartesian_sampling_points[:, -1, :]

            # cartesian pos (and orn) of the last point of the action spline
            # (to compute a reward based on the cartesian distance to the target point if the robot base is floating)
            self._spline_last_cartesian_action_point = action_spline_cartesian_sampling_points[:, -1, :]

    def get_total_spline_deviation(self):
        total_spline_sampling_interpolation = self.total_spline.get_interpolation(
            sample_distance=self._spline_deviation_sample_distance,
            use_normalized_sample_distance=False, convert_to_u=False, lin_space=True)[1:]

        reference_spline_sampling_interpolation = \
            total_spline_sampling_interpolation + self._reference_spline.start_dis
        sampling_indices_at_reference_spline_end = \
            reference_spline_sampling_interpolation >= self._reference_spline.end_dis
        sampling_indices_not_at_reference_spline_end = np.invert(sampling_indices_at_reference_spline_end)
        reference_spline_sampling_interpolation[sampling_indices_at_reference_spline_end] = \
            self._reference_spline.end_dis

        total_spline_sampling_points = self.total_spline.evaluate_length(total_spline_sampling_interpolation).T
        reference_spline_sampling_points = np.zeros_like(total_spline_sampling_points)

        if np.any(sampling_indices_at_reference_spline_end):
            reference_spline_sampling_points[sampling_indices_at_reference_spline_end, :] = \
                self._reference_spline.curve_data_spline[:, self._reference_spline.u_end_index]

        if np.any(sampling_indices_not_at_reference_spline_end):
            reference_spline_sampling_points[sampling_indices_not_at_reference_spline_end, :] = \
                self._reference_spline.evaluate_length(
                    reference_spline_sampling_interpolation[sampling_indices_not_at_reference_spline_end]).T

        total_spline_deviation = np.linalg.norm(reference_spline_sampling_points - total_spline_sampling_points, axis=1)

        total_spline_mean_deviation = np.mean(total_spline_deviation)
        total_spline_max_deviation = np.max(total_spline_deviation)
        total_spline_final_deviation = total_spline_deviation[-1]

        total_spline_cartesian_sampling_points = \
            convert_joint_space_to_cartesian_space(self._env, joint_data=total_spline_sampling_points.T)
        reference_spline_cartesian_sampling_points = np.zeros_like(total_spline_cartesian_sampling_points)
        if np.any(sampling_indices_at_reference_spline_end):
            reference_spline_cartesian_sampling_points[:, sampling_indices_at_reference_spline_end, :] = \
                convert_joint_space_to_cartesian_space(
                    self._env,
                    joint_data=self._reference_spline.curve_data_spline[:, self._reference_spline.u_end_index])
        if np.any(sampling_indices_not_at_reference_spline_end):
            reference_spline_cartesian_sampling_points[:, sampling_indices_not_at_reference_spline_end, :] = \
                convert_joint_space_to_cartesian_space(
                    self._env,
                    joint_data=reference_spline_sampling_points[sampling_indices_not_at_reference_spline_end, :].T)

        # pos deviation
        total_spline_cartesian_pos_deviation = np.linalg.norm(total_spline_cartesian_sampling_points[:, :, 0:3] -
                                                          reference_spline_cartesian_sampling_points[:, :, 0:3], axis=2)

        total_spline_mean_cartesian_pos_deviation = np.mean(total_spline_cartesian_pos_deviation)
        total_spline_max_cartesian_pos_deviation = np.max(total_spline_cartesian_pos_deviation)
        total_spline_final_cartesian_pos_deviation = np.mean(total_spline_cartesian_pos_deviation[:, -1])

        # orn deviation
        # the difference between two unit quaternions can be described as a single rotation angle in [0, pi]
        # using the following formula: 2 * arccos(abs(<q_1, q_2>)) with <> being the dot product

        total_spline_cartesian_orn_deviation_rad = []
        for i in range(len(total_spline_cartesian_sampling_points)):
            dot_product = np.einsum('ij,ij->i', total_spline_cartesian_sampling_points[i, :, 3:],
                                    reference_spline_cartesian_sampling_points[i, :, 3:])
            # the einsum is equivalent to a matrix multiplication where only the diagonal elements are computed.
            orn_deviation_rad = 2 * np.arccos(np.minimum(np.abs(dot_product), 1.0))
            total_spline_cartesian_orn_deviation_rad.append(orn_deviation_rad)
        total_spline_cartesian_orn_deviation_rad = np.asarray(total_spline_cartesian_orn_deviation_rad)

        total_spline_mean_cartesian_orn_deviation_rad = np.mean(total_spline_cartesian_orn_deviation_rad)
        total_spline_max_cartesian_orn_deviation_rad = np.max(total_spline_cartesian_orn_deviation_rad)
        total_spline_final_cartesian_orn_deviation_rad = np.mean(total_spline_cartesian_orn_deviation_rad[:, -1])

        return total_spline_mean_deviation, total_spline_max_deviation, total_spline_final_deviation, \
            total_spline_mean_cartesian_pos_deviation, total_spline_max_cartesian_pos_deviation, \
            total_spline_final_cartesian_pos_deviation, total_spline_mean_cartesian_orn_deviation_rad, \
            total_spline_max_cartesian_orn_deviation_rad, total_spline_final_cartesian_orn_deviation_rad

    def get_final_spline_deviation(self):
        if self._spline_deviation_weighting_factors is not None:
            weighted_deviation = (self._reference_spline.curve_data_spline[:,
                                  self._reference_spline.u_end_index] -
                                  self._action_spline.curve_data_spline[:,
                                  self._action_spline.u_end_index]) \
                                 * self._spline_deviation_weighting_factors
            final_spline_deviation = self._spline_deviation_normalizing_factor * \
                np.linalg.norm(weighted_deviation, axis=0)
        else:
            final_spline_deviation = np.linalg.norm(self._reference_spline.curve_data_spline[:,
                                                    self._reference_spline.u_end_index] -
                                                    self._action_spline.curve_data_spline[:,
                                                    self._action_spline.u_end_index], axis=0)
        return final_spline_deviation

    def get_trajectory_start_position(self):
        return self._trajectory_start_position

    def get_generated_trajectory_point(self, index, key='positions'):
        i = clip_index(index, len(self._generated_trajectory[key]))

        return self._generated_trajectory[key][i]

    def get_measured_actual_trajectory_control_point(self, index, start_at_index=False, key='positions'):
        i = clip_index(index, len(self._measured_actual_trajectory_control_points[key]))

        if not start_at_index:
            return self._measured_actual_trajectory_control_points[key][i]
        else:
            return self._measured_actual_trajectory_control_points[key][i:]

    def get_computed_actual_trajectory_control_point(self, index, start_at_index=False, key='positions'):
        i = clip_index(index, len(self._computed_actual_trajectory_control_points[key]))

        if not start_at_index:
            return self._computed_actual_trajectory_control_points[key][i]
        else:
            return self._computed_actual_trajectory_control_points[key][i:]

    def get_generated_trajectory_control_point(self, index, key='positions'):
        i = clip_index(index, len(self._generated_trajectory_control_points[key]))

        return self._generated_trajectory_control_points[key][i]

    def add_generated_trajectory_point(self, positions, velocities, accelerations):
        self._generated_trajectory['positions'].append(positions)
        self._generated_trajectory['velocities'].append(velocities)
        self._generated_trajectory['accelerations'].append(accelerations)

    def add_measured_actual_trajectory_control_point(self, positions, velocities, accelerations):
        self._measured_actual_trajectory_control_points['positions'].append(positions)
        self._measured_actual_trajectory_control_points['velocities'].append(velocities)
        self._measured_actual_trajectory_control_points['accelerations'].append(accelerations)

    def add_computed_actual_trajectory_control_point(self, positions, velocities, accelerations):
        self._computed_actual_trajectory_control_points['positions'].append(positions)
        self._computed_actual_trajectory_control_points['velocities'].append(velocities)
        self._computed_actual_trajectory_control_points['accelerations'].append(accelerations)

    def add_generated_trajectory_control_point(self, positions, velocities, accelerations):
        self._generated_trajectory_control_points['positions'].append(positions)
        self._generated_trajectory_control_points['velocities'].append(velocities)
        self._generated_trajectory_control_points['accelerations'].append(accelerations)

    def compute_controller_model_coefficients(self, time_constants, sampling_time):
        self._controller_model_coefficient_a = 1 + (2 * np.array(time_constants) / sampling_time)
        self._controller_model_coefficient_b = 1 - (2 * np.array(time_constants) / sampling_time)

    def model_position_controller_to_compute_actual_values(self, current_setpoint, last_setpoint, key='positions'):
        # models the position controller as a discrete transfer function and returns the
        # computed actual position, given the next position setpoint and previous computed actual positions
        # the controller is modelled as a first order low-pass with a (continuous) transfer function of
        #  G(s) = 1 / (1 + T * s)
        # the transfer function is discretized using Tustins approximation: s = 2 / Ta * (z - 1) / (z + 1)
        # the following difference equation can be derived:
        # y_n = 1/a * (x_n + x_n_minus_one - b * y_n_minus_one) with a = 1 + (2 * T / Ta) and b = 1 - (2 * T / Ta)

        x_n = np.asarray(current_setpoint)
        x_n_minus_one = np.asarray(last_setpoint)
        y_n_minus_one = self.get_computed_actual_trajectory_control_point(-1, key=key)
        computed_actual_position = 1 / self._controller_model_coefficient_a * \
                                   (x_n + x_n_minus_one - self._controller_model_coefficient_b * y_n_minus_one)

        return computed_actual_position

    def is_trajectory_finished(self, index):
        if self._use_splines:
            # current reference_spline distance has reached the end for a specified number of time steps
            if self._spline_termination_extra_time_steps is not None \
                    and self._spline_extra_time_steps >= self._spline_termination_extra_time_steps:
                return True, TERMINATION_SPLINE_LENGTH

            # max deviation exceeds limits
            if self._spline_termination_max_deviation is not None \
                    and self.spline_max_deviation > self._spline_termination_max_deviation:
                return True, TERMINATION_SPLINE_DEVIATION

        if index >= self._trajectory_length - 1:  # trajectory duration
            return True, TERMINATION_TRAJECTORY_LENGTH

        return False, None

    def _get_new_trajectory_start_position(self):
        return self._obstacle_wrapper.get_starting_point_joint_pos()




