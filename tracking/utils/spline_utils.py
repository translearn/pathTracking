import errno
import json
import math
import os
import time

import numpy as np
import pybullet as p
import logging
from scipy import integrate
from scipy.interpolate import splprep, splev, make_interp_spline, BSpline


class Spline:
    def __init__(self,
                 orig=None,
                 curve_data=None,
                 curve_data_slicing_step=1,
                 length_correction_step_size=0.01,
                 use_normalized_length_correction_step_size=True,
                 method="auto",  # possible options: "auto", "bspline", "fitpack"
                 curvature_at_ends=None):  # enforce the second derivative to have a specified value

        self._reflection_vectors = None
        self._reflection_vector_index = 0

        if orig is None:
            # if curve_data is None, an "empty" object is created (e.g. for loading from json)
            if curve_data is not None:
                # requires curve_data in the form
                # [dimension e.g. joint index or Cartesian dimension][data point index j]
                self._length_correction_step_size = length_correction_step_size
                self._use_normalized_length_correction_step_size = use_normalized_length_correction_step_size
                self._curve_data_slicing_step = curve_data_slicing_step
                self._method = method
                self._curvature_at_ends = curvature_at_ends
                self._num_dimensions = len(curve_data)
                available_methods = ["auto", "bspline", "fitpack"]
                if method in available_methods:
                    if method == "auto":
                        method = "bspline"

                    if curve_data_slicing_step != 1:
                        curve_data = curve_data[:, ::self._curve_data_slicing_step]

                    # subsequent knots are not allowed to be equal as the spline computation will fail otherwise
                    self._u = get_u(curve_data=curve_data)
                    if self._u is not None:
                        valid_u = np.concatenate(([True], np.invert(np.equal(self._u[:-1], self._u[1:]))))
                        self._u = self._u[valid_u]
                        curve_data = curve_data[:, valid_u]
                    else:
                        curve_data = curve_data[:, 0:1]  # self._u is None -> length is zero -> all elements are equal

                    if curve_data.shape[1] == 1:
                        # a spline cannot be constructed from a single knot
                        # -> add a second knot that is almost identical
                        curve_data = np.concatenate((curve_data,
                                                    curve_data + 1e-6), axis=1)
                        self._u = get_u(curve_data=curve_data)

                    self._curve_data_spline = curve_data

                    if method == "bspline":
                        if self._curvature_at_ends is None:
                            bc_type = None
                        elif self._curvature_at_ends == 0.0:
                            bc_type = "natural"
                        else:
                            # curvature at ends has to be a tuple with values for each dimension ([], []) or a scalar
                            # for all dimensions (x, y)
                            curvature_left, curvature_right = self._curvature_at_ends
                            if type(curvature_left) in [int, float]:
                                curvature_left = [curvature_left] * self._num_dimensions
                            if type(curvature_right) in [int, float]:
                                curvature_right = [curvature_right] * self._num_dimensions
                            bc_type = ([(2, curvature_left)], [(2, curvature_right)])

                        try:
                            self._b_spline = make_interp_spline(self._u, self._curve_data_spline.T,
                                                                bc_type=bc_type)
                        except Exception as e:
                            raise ValueError("Error in b_spline calculation: message {}, u {}, "
                                             "curve_data_spline {}".format(e, self._u, self._curve_data_spline))
                        self._tck = None
                    else:
                        if self._curvature_at_ends is not None:
                            raise ValueError("curvature_at_ends!=None requires method==bspline")
                        self._tck, self._u = splprep(x=self._curve_data_spline, s=0)
                        self._b_spline = None

                    self._uncorrected_curve_length = None
                    self._u_length = self._compute_u_length()

                else:
                    raise ValueError("method must be on of the following values: {}".format(available_methods))
        else:
            self._copy_attributes(orig=orig)

        self._u_arc = None
        self._u_dis = None
        self._u_arc_diff = None

        self._u_start = None
        self._u_end_index = None
        self._start_dis = None
        self._end_dis = None
        self._current_dis = None

        self.reset()

    def reset(self, reflection_vector_index=0, random_reflection_vector_index=False):
        self._u_start = 0
        self._u_end_index = None
        self._start_dis = None
        self._end_dis = None
        self._current_dis = 0

        if self._reflection_vectors is not None:
            if random_reflection_vector_index:
                reflection_vector_index = np.random.randint(0, len(self._reflection_vectors))
            if self._reflection_vector_index != reflection_vector_index:
                self._reflection_vector_index = reflection_vector_index
                self._curve_data_spline = None

    def _copy_attributes(self, orig):
        self._length_correction_step_size = orig._length_correction_step_size
        self._use_normalized_length_correction_step_size = orig._use_normalized_length_correction_step_size
        self._curve_data_slicing_step = orig._curve_data_slicing_step
        self._curve_data_spline = orig._curve_data_spline
        self._u = orig._u
        self._b_spline = orig._b_spline
        self._tck = orig._tck
        self._u_length = orig._u_length
        self._method = orig._method
        self._curvature_at_ends = orig._curvature_at_ends
        self._uncorrected_curve_length = orig._uncorrected_curve_length

    def copy_with_adjusted_length_correction(self, length_correction_step_size=0.01,
                                             use_normalized_length_correction_step_size=True):
        adjusted_spline = Spline(orig=self)
        adjusted_spline._length_correction_step_size = length_correction_step_size
        adjusted_spline._use_normalized_length_correction_step_size = use_normalized_length_correction_step_size
        adjusted_spline._u_length = adjusted_spline._compute_u_length()
        return adjusted_spline

    def copy_with_resampling(self, resampling_distance, use_normalized_resampling_distance=True,
                             use_curvature_for_resampling=False,
                             length_correction_step_size=-1, use_normalized_length_correction_step_size=-1,
                             curvature_at_ends=-1):

        if length_correction_step_size == -1:
            length_correction_step_size = self._length_correction_step_size
        if use_normalized_length_correction_step_size == -1:
            use_normalized_length_correction_step_size = self._use_normalized_length_correction_step_size
        if curvature_at_ends == -1:
            curvature_at_ends = self._curvature_at_ends

        if use_curvature_for_resampling:
            interpolation = self.get_interpolation(sample_distance=length_correction_step_size,
                                                   use_normalized_sample_distance=
                                                   use_normalized_length_correction_step_size,
                                                   convert_to_u=False)
            if not use_normalized_length_correction_step_size:
                u_arc_interpolation = interpolation / self.get_length()
            else:
                u_arc_interpolation = interpolation
            u_interpolation = self.u_arc_to_u(u_arc_interpolation)
            interpolated_pos = self.evaluate(u=u_interpolation)
            interpolated_length = u_arc_interpolation * self.get_length()
            interpolated_length_diff = interpolated_length[1] - interpolated_length[0]
            interpolated_der_1 = np.diff(interpolated_pos, axis=1) / interpolated_length_diff
            interpolated_der_2 = np.diff(interpolated_der_1, axis=1) / interpolated_length_diff
            interpolated_der_2_norm = np.linalg.norm(interpolated_der_2, axis=0)
            interpolated_der_2_norm_integration = np.cumsum(interpolated_der_2_norm) * interpolated_length_diff
            interpolated_der_2_norm_integration = np.concatenate(([0.0], interpolated_der_2_norm_integration,
                                                                  [2 * interpolated_der_2_norm_integration[-1]
                                                                   - interpolated_der_2_norm_integration[-2]]))
            u_curvature_integration = np.stack((u_interpolation,
                                                interpolated_der_2_norm_integration))

            length = None if use_normalized_resampling_distance else get_length(u_curvature_integration)

            interpolation = self.get_interpolation(sample_distance=resampling_distance,
                                                   use_normalized_sample_distance=
                                                   use_normalized_resampling_distance,
                                                   length=length,
                                                   convert_to_u=False,
                                                   lin_space=False)
            if use_normalized_resampling_distance:
                curvature_interpolation = interpolation * get_length(u_curvature_integration)
            else:
                curvature_interpolation = interpolation
            u_interpolation = length_to_u(curvature_interpolation, u_curvature_integration)
        else:
            u_interpolation = self.get_interpolation(sample_distance=resampling_distance,
                                                     use_normalized_sample_distance=
                                                     use_normalized_resampling_distance,
                                                     convert_to_u=True,
                                                     lin_space=False)

        curve_data_resampling = self.evaluate(u=u_interpolation)

        resampled_spline = Spline(curve_data=curve_data_resampling,
                                  length_correction_step_size=length_correction_step_size,
                                  curve_data_slicing_step=1,
                                  use_normalized_length_correction_step_size=
                                  use_normalized_length_correction_step_size,
                                  method=self._method,
                                  curvature_at_ends=curvature_at_ends)
        return resampled_spline

    def evaluate(self, u):

        if self._b_spline is None:
            spline_data = np.array(splev(u, self._tck))
        else:
            spline_data = np.array(self._b_spline(u)).T

        if self._reflection_vectors is None:
            return spline_data
        else:
            return (spline_data.T * self._reflection_vectors[self._reflection_vector_index]).T

    def evaluate_u_arc(self, u_arc):
        return self.evaluate(u=self.u_arc_to_u(u_arc))

    def evaluate_length(self, length):
        return self.evaluate(u=self.length_to_u(length))

    def set_u_start(self, u_start=None, u_arc_start=None):
        # also sets current_dis to the dis corresponding to u_start
        if u_arc_start is not None:
            self._u_start = self.u_arc_to_u(u_arc_start)
            self._current_dis = self.u_arc_to_length(u_arc_start)
        else:
            self._u_start = u_start
            self._current_dis = self.u_to_length(u_start)

    def set_u_end_index(self, u_end_min=None, u_arc_end_min=None):
        if u_end_min is None:
            u_end_min = self.u_arc_to_u(u_arc_end_min)

        if u_end_min != 1.0:
            u_end_index = np.searchsorted(self._u, u_end_min)
        else:
            u_end_index = len(self._u) - 1

        self._u_end_index = u_end_index

    def add_dis_to_current_dis(self, dis):
        self._current_dis = self._current_dis + dis
        is_finished = False
        if self._current_dis >= self.end_dis:
            self._current_dis = self.end_dis
            is_finished = True

        return is_finished

    def get_current_knots(self, n_next=1):
        # returns the curve data of the current knots (knot before self._current_dis n_next following knots) and the
        # distance between the knots
        current_knot_index = np.searchsorted(self.u_dis, self._current_dis)
        if self.u_dis[current_knot_index] > self._current_dis:
            current_knot_index = current_knot_index - 1
        current_knots_indices = np.minimum(np.arange(current_knot_index, current_knot_index + n_next + 1),
                                           self.u_end_index)
        current_knots_curve_data = self.curve_data_spline.T[current_knots_indices]
        current_knots_dis = self.u_dis[current_knots_indices]

        return current_knots_curve_data, current_knots_dis, current_knots_indices

    def get_joint_position(self, u=None, u_arc=None):
        if u_arc is not None:
            return self.evaluate_u_arc(u_arc)
        else:
            return self.evaluate(u)

    def _compute_u_length(self):
        u_length = None
        if self._length_correction_step_size is None or not self._use_normalized_length_correction_step_size:
            # self._curve_data_spline -> knots used to generate the spline
            # in the form [dimension e.g. joint index or Cartesian dimension][data point index j]
            # calculate the total length of the spline assuming linear segments between the knots
            if self._uncorrected_curve_length is None:
                self._uncorrected_curve_length = \
                    get_curve_length(curve_data=self.curve_data_spline, cumulated_sum=False)
            u_length = self._uncorrected_curve_length

        if self._length_correction_step_size is not None:
            if self._use_normalized_length_correction_step_size:
                num_u = round(1 / self._length_correction_step_size) + 1
            else:
                num_u = max(round(u_length / self._length_correction_step_size) + 1, 2)
            u_interpolation = np.linspace(0, 1, num_u)
            interpolated_spline_data = self.evaluate(u_interpolation)
            u_length = np.stack((u_interpolation,
                                 get_curve_length(curve_data=interpolated_spline_data, cumulated_sum=True)))

            if u_length[1][-1] == 0:
                u_length[1][-1] = 1e-6

        return u_length

    def u_to_length(self, u):
        return u_to_length(u, self._u_length)

    def u_arc_to_u(self, u_arc):
        return u_arc_to_u(u_arc, self._u_length)

    def u_to_u_arc(self, u):
        return u_to_u_arc(u, self._u_length)

    def u_arc_to_length(self, u_arc):
        return u_arc_to_length(u_arc, self._u_length)

    def length_to_u(self, length):
        return length_to_u(length, self._u_length)

    def get_length(self):
        return get_length(self._u_length)

    def get_interpolation(self, sample_distance, use_normalized_sample_distance, length=None, convert_to_u=False,
                          u_start=None, u_end=None,
                          lin_space=True):
        if length is None:
            if not use_normalized_sample_distance:
                length = self.get_length()
        else:
            if use_normalized_sample_distance or u_end is not None:
                logging.info("Ignored attribute length!")

        interpolation_min = 0
        if u_start is not None:
            if use_normalized_sample_distance:
                interpolation_min = self.u_to_u_arc(u_start)
            else:
                interpolation_min = self.u_to_length(u_start)

        if u_end is not None:
            if use_normalized_sample_distance:
                interpolation_max = self.u_to_u_arc(u_end)
            else:
                interpolation_max = self.u_to_length(u_end)
        else:
            interpolation_max = 1.0 if use_normalized_sample_distance else length

        if lin_space:
            num_interpolation = max(round((interpolation_max - interpolation_min) / sample_distance) + 1, 2)
            interpolation = np.linspace(interpolation_min, interpolation_max, num_interpolation)
        else:
            interpolation = np.concatenate((np.arange(interpolation_min, interpolation_max, sample_distance),
                                           [interpolation_max]))

        if convert_to_u:
            if use_normalized_sample_distance:
                return self.u_arc_to_u(interpolation)
            else:
                return self.length_to_u(interpolation)
        else:
            return interpolation

    @property
    def u(self):
        return self._u

    @property
    def u_arc(self):
        if self._u_arc is None:
            self._u_arc = self.u_to_length(self._u) / self.get_length()
        return self._u_arc

    @property
    def current_dis(self):
        return self._current_dis

    @property
    def u_dis(self):
        if self._u_dis is None:
            self._u_dis = self.u_to_length(self._u)
        return self._u_dis

    @property
    def u_start(self):
        return self._u_start

    @property
    def start_dis(self):
        if self._start_dis is None:
            self._start_dis = self.u_to_length(self._u_start)
        return self._start_dis

    @property
    def u_end_index(self):
        if self._u_end_index is None:
            self._u_end_index = len(self._u) - 1
        return self._u_end_index

    @property
    def end_dis(self):
        if self._end_dis is None:
            self._end_dis = self.u_dis[self.u_end_index]
        return self._end_dis

    @property
    def u_arc_diff(self):
        if self._u_arc_diff is None:
            self._u_arc_diff = np.diff(self.u_arc)
        return self._u_arc_diff

    @property
    def max_dis_between_knots(self):
        return self.u_arc_to_length(np.max(self.u_arc_diff))

    @property
    def curve_data_spline(self):
        if self._curve_data_spline is None:  # curve_data_spline can always be reconstructed by sampling
            # -> no need to store curve_data_spline when saving the spline to a json file
            self._curve_data_spline = self.evaluate(u=self._u)
        return self._curve_data_spline

    def save_to_json(self, path, make_dir=False):
        spline_dict = {'_length_correction_step_size': self._length_correction_step_size,
                       '_use_normalized_length_correction_step_size': self._use_normalized_length_correction_step_size,
                       '_curve_data_slicing_step': self._curve_data_slicing_step,
                       '_u': self._u.tolist(),
                       '_method': self._method,
                       '_curvature_at_ends': self._curvature_at_ends,
                       '_uncorrected_curve_length': self._uncorrected_curve_length,
                       }

        if self._b_spline is not None:
            spline_dict['_b_spline'] = dict(t=self._b_spline.t.tolist(), c=self._b_spline.c.tolist(),
                                            k=self._b_spline.k)
            spline_dict['_tck'] = None
        else:
            spline_dict['_b_spline'] = None
            spline_dict['_tck'] = dict(t=self._tck[0].tolist(), c=np.array(self._tck[1]).tolist(),
                                       k=self._tck[2])

        if make_dir:
            save_dir = os.path.dirname(path)
            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir)
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

        with open(path, 'w') as f:
            f.write(json.dumps(spline_dict))
            f.flush()

    def visualize(self, env, sample_distance=None, use_normalized_sample_distance=True,
                  debug_line_buffer=None, append_spline=False, visualize_knots=True, visualize_knots_orn=False,
                  line_width=2.0, debug_line_buffer_init_size=500, u_start=None, u_end=None,
                  line_color=None, u_marker=None, marker_color=None, marker_width=None, physics_client_id=None):
        u_start = self._u_start if u_start is None else u_start

        if u_start != 0:
            u_start_index = np.searchsorted(self._u, u_start)
        else:
            u_start_index = 0

        if u_end is not None:
            if u_end != 1.0:
                u_end_index = np.searchsorted(self._u, u_end)
            else:
                u_end_index = len(self._u) - 1
        else:
            u_end_index = self.u_end_index
            u_end = self._u[u_end_index]

        curve_data_visualization_equals_knots = True

        if sample_distance is None:
            curve_data_visualization = self.curve_data_spline[:, u_start_index:u_end_index + 1]
            # use knots of the spline
            if u_start != 0:
                if self._u[u_start_index] > u_start:
                    curve_data_visualization_equals_knots = False
                    curve_data_u_start = self.evaluate(u=u_start)
                    curve_data_u_start = curve_data_u_start.reshape(-1, 1)
                    curve_data_visualization = np.concatenate((curve_data_u_start,
                                                               curve_data_visualization), axis=1)

            if self._u[u_end_index] > u_end:
                curve_data_visualization_equals_knots = False
                u_end_index = u_end_index - 1
                curve_data_visualization = curve_data_visualization[:, :-1]
                curve_data_u_end = self.evaluate(u=u_end)
                curve_data_u_end = curve_data_u_end.reshape(-1, 1)
                curve_data_visualization = np.concatenate((curve_data_visualization,
                                                           curve_data_u_end), axis=1)

        else:
            curve_data_visualization_equals_knots = False
            if not use_normalized_sample_distance:
                u_start_interpolation = self.length_to_u(np.ceil(self.u_to_length(u_start) / sample_distance)
                                                         * sample_distance)
            else:
                u_start_interpolation = u_start
            u_interpolation = self.get_interpolation(sample_distance=sample_distance,
                                                     use_normalized_sample_distance=use_normalized_sample_distance,
                                                     u_start=u_start_interpolation,
                                                     u_end=u_end,
                                                     convert_to_u=True,
                                                     lin_space=False)
            curve_data_visualization = self.evaluate(u=u_interpolation)

        if physics_client_id is None:
            physics_client_id = env._obstacle_client_id

        cartesian_data_visualization = convert_joint_space_to_cartesian_space(env, joint_data=curve_data_visualization)

        if visualize_knots:
            if curve_data_visualization_equals_knots:
                cartesian_data_knots = cartesian_data_visualization
            else:
                cartesian_data_knots = \
                    convert_joint_space_to_cartesian_space(env,
                                                           joint_data=
                                                           self.curve_data_spline[:, u_start_index:u_end_index + 1])
        else:
            cartesian_data_knots = None

        if u_marker is not None:
            # add marker at a specific position specified by u
            if type(u_marker) == float:
                u_marker = [u_marker]
            marker_data = self.evaluate(u=u_marker)
            cartesian_marker_data = \
                convert_joint_space_to_cartesian_space(env, joint_data=marker_data)
        else:
            cartesian_marker_data = None

        if debug_line_buffer is None:
            debug_line_buffer = [[], -1]  # [list of userDataIds, userDataIds in use]
            if debug_line_buffer_init_size is not None:
                # prevent debug line bug by generating debug lines in advance and replacing them afterwards
                # (the bug does not occur when replacing debug lines)
                debug_line_buffer = \
                    self.init_debug_line_buffer(debug_line_buffer=debug_line_buffer,
                                                debug_line_buffer_init_size=
                                                len(cartesian_data_visualization) * debug_line_buffer_init_size,
                                                physics_client_id=physics_client_id)

        if append_spline:
            debug_line_buffer_index = debug_line_buffer[1]
        else:
            debug_line_buffer_index = -1

        custom_line_color = True if line_color is not None else False

        for i in range(len(cartesian_data_visualization)):
            # for each target link point
            line_color = env._robot_scene.obstacle_wrapper.get_target_point_color(robot=i)[0:3] \
                if not custom_line_color else line_color
            for j in range(len(cartesian_data_visualization[i]) - 1):
                # for each cartesian data point ([0:3] -> pos)
                debug_line_buffer_index = debug_line_buffer_index + 1
                add_debug_line(line_from=cartesian_data_visualization[i][j][0:3],
                               line_to=cartesian_data_visualization[i][j + 1][0:3],
                               debug_line_buffer=debug_line_buffer,
                               debug_line_buffer_index=debug_line_buffer_index,
                               line_color=line_color,
                               line_width=line_width,
                               physics_client_id=physics_client_id)

            if visualize_knots:
                if visualize_knots_orn:
                    tick_length = 0.05
                    x_tick = [tick_length, 0, 0]
                    y_tick = [0, tick_length, 0]
                    z_tick = [0, 0, tick_length]
                    xyz_ticks = [x_tick, y_tick, z_tick]
                    default_orn = p.getQuaternionFromEuler([0, 0, 0])
                    line_width = 1
                else:
                    tick_length = 0.01
                    tick_color = line_color

                for j in range(len(cartesian_data_knots[i])):

                    if visualize_knots_orn:
                        for k in range(len(xyz_ticks)):
                            debug_line_buffer_index = debug_line_buffer_index + 1
                            tick_rotated, _ = p.multiplyTransforms(positionA=[0, 0, 0],
                                                                   orientationA=cartesian_data_knots[i][j][3:],
                                                                   positionB=xyz_ticks[k],
                                                                   orientationB=default_orn)

                            line_color = [0, 0, 0]
                            line_color[k] = 1.0
                            add_debug_line(line_from=[cartesian_data_knots[i][j][0],
                                                      cartesian_data_knots[i][j][1],
                                                      cartesian_data_knots[i][j][2]],
                                           line_to=[cartesian_data_knots[i][j][0] + tick_rotated[0],
                                                    cartesian_data_knots[i][j][1] + tick_rotated[1],
                                                    cartesian_data_knots[i][j][2] + tick_rotated[2]],
                                           debug_line_buffer=debug_line_buffer,
                                           debug_line_buffer_index=debug_line_buffer_index,
                                           line_color=line_color,
                                           line_width=line_width,
                                           physics_client_id=physics_client_id)
                    else:
                        debug_line_buffer_index = debug_line_buffer_index + 1
                        add_debug_line(line_from=[cartesian_data_knots[i][j][0],
                                                  cartesian_data_knots[i][j][1],
                                                  cartesian_data_knots[i][j][2] - tick_length],
                                       line_to=[cartesian_data_knots[i][j][0],
                                                cartesian_data_knots[i][j][1],
                                                cartesian_data_knots[i][j][2] + tick_length],
                                       debug_line_buffer=debug_line_buffer,
                                       debug_line_buffer_index=debug_line_buffer_index,
                                       line_color=tick_color,
                                       line_width=line_width,
                                       physics_client_id=physics_client_id)

            if cartesian_marker_data is not None:
                tick_length = 0.0075
                marker_color = line_color if marker_color is None else marker_color
                marker_width = line_width if marker_width is None else marker_width
                for j in range(len(cartesian_marker_data[i])):
                    debug_line_buffer_index = debug_line_buffer_index + 1
                    add_debug_line(line_from=[cartesian_marker_data[i][j][0],
                                              cartesian_marker_data[i][j][1] - tick_length,
                                              cartesian_marker_data[i][j][2] - tick_length],
                                   line_to=[cartesian_marker_data[i][j][0],
                                            cartesian_marker_data[i][j][1] + tick_length,
                                            cartesian_marker_data[i][j][2] + tick_length],
                                   debug_line_buffer=debug_line_buffer,
                                   debug_line_buffer_index=debug_line_buffer_index,
                                   line_color=marker_color,
                                   line_width=marker_width,
                                   physics_client_id=physics_client_id)
                    debug_line_buffer_index = debug_line_buffer_index + 1
                    add_debug_line(line_from=[cartesian_marker_data[i][j][0],
                                              cartesian_marker_data[i][j][1] - tick_length,
                                              cartesian_marker_data[i][j][2] + tick_length],
                                   line_to=[cartesian_marker_data[i][j][0],
                                            cartesian_marker_data[i][j][1] + tick_length,
                                            cartesian_marker_data[i][j][2] - tick_length],
                                   debug_line_buffer=debug_line_buffer,
                                   debug_line_buffer_index=debug_line_buffer_index,
                                   line_color=marker_color,
                                   line_width=marker_width,
                                   physics_client_id=physics_client_id)

        for i in range(debug_line_buffer_index + 1, debug_line_buffer[1] + 1):
            p.addUserDebugLine([0, 0, 0],
                               [0, 0, 0],
                               replaceItemUniqueId=debug_line_buffer[0][i],
                               physicsClientId=physics_client_id)

        debug_line_buffer[1] = debug_line_buffer_index

        return debug_line_buffer

    @staticmethod
    def reset_debug_line_buffer(debug_line_buffer, physics_client_id):
        if debug_line_buffer is not None:
            for i in range(0, debug_line_buffer[1] + 1):
                p.addUserDebugLine([0, 0, 0],
                                   [0, 0, 0],
                                   replaceItemUniqueId=debug_line_buffer[0][i],
                                   physicsClientId=physics_client_id)
            debug_line_buffer[1] = -1
        return debug_line_buffer

    @classmethod
    def load_from_json(cls, path, length_correction_step_size=None, use_normalized_length_correction_step_size=None,
                       reflection_vectors=None):
        # recompute _u_length as much storage is required otherwise
        with open(path) as file:
            spline_dict = json.load(file)
        return Spline.load_from_dict(spline_dict, length_correction_step_size=length_correction_step_size,
                                     use_normalized_length_correction_step_size=
                                     use_normalized_length_correction_step_size,
                                     reflection_vectors=reflection_vectors)

    @classmethod
    def load_from_dict(cls, spline_dict, length_correction_step_size=None,
                       use_normalized_length_correction_step_size=None,
                       reflection_vectors=None):
        # recompute _u_length as much storage is required otherwise

        spline = cls()
        spline._length_correction_step_size = spline_dict['_length_correction_step_size'] \
            if length_correction_step_size is None else length_correction_step_size
        spline._use_normalized_length_correction_step_size = \
            spline_dict['_use_normalized_length_correction_step_size'] \
            if use_normalized_length_correction_step_size is None else use_normalized_length_correction_step_size
        spline._curve_data_slicing_step = spline_dict['_curve_data_slicing_step']
        spline._u = np.array(spline_dict['_u'])
        spline._method = spline_dict['_method']
        spline._curvature_at_ends = spline_dict['_curvature_at_ends']

        spline._uncorrected_curve_length = spline_dict['_uncorrected_curve_length']
        spline._curve_data_spline = None  # reconstruct by accessing curve_data_spline if desired
        if reflection_vectors is not None:
            spline._reflection_vectors = np.asarray(reflection_vectors)

        if spline_dict['_b_spline']:
            spline._b_spline = BSpline.construct_fast(t=np.array(spline_dict['_b_spline']['t']),
                                                      c=np.array(spline_dict['_b_spline']['c']),
                                                      k=spline_dict['_b_spline']['k'])
            spline._tck = None
        else:
            spline._b_spline = None
            spline._tck = [np.array(spline_dict['_tck']['t']),
                           list(np.array(spline_dict['_tck']['c'])),
                           spline_dict['_tck']['k']]

        spline._u_length = spline._compute_u_length()
        return spline

    @staticmethod
    def init_debug_line_buffer(debug_line_buffer, debug_line_buffer_init_size, physics_client_id):
        valid_debug_lines = 0
        while valid_debug_lines < debug_line_buffer_init_size:
            debug_line_id = p.addUserDebugLine([0, 0, 0],
                                               [0, 0, 0],
                                               physicsClientId=physics_client_id)
            if debug_line_id != -1:
                valid_debug_lines = valid_debug_lines + 1
                debug_line_buffer[0].append(debug_line_id)
        return debug_line_buffer


def get_curve_length(curve_data, cumulated_sum=False):
    # requires curve_data in the form [dimension e.g. joint index or Cartesian dimension][data point index j]
    # calculate length of a curve by integration
    # assuming linear line segments between data points -> length of line segments corresponds to the norm of
    # the difference between two data points
    # Note: This assumption requires dense sampling. Otherwise, the shape of the curve between two data points
    # has to be considered, e.g. by numerical integration
    curve_data_delta = np.diff(curve_data, axis=1)
    curve_data_delta_norm = np.linalg.norm(curve_data_delta, axis=0)
    return np.concatenate(([0.0], np.cumsum(curve_data_delta_norm))) if cumulated_sum \
        else float(np.sum(curve_data_delta_norm))


def get_u(curve_data):
    cumulated_curve_length = get_curve_length(curve_data, cumulated_sum=True)
    if cumulated_curve_length[-1] != 0:
        u = cumulated_curve_length / cumulated_curve_length[-1]
    else:
        u = None
    return u


def u_to_length(u, u_length):
    # u_length[0][x] -> u values, u_length[1][x] -> corresponding length values
    # computes the corresponding length to an array of u_desired values in [0.0, 1.0]
    if type(u) == list:
        u = np.array(u)
    if type(u_length) == float:
        # assuming line segments between the knots, the relation is linear
        length = u * u_length
    else:
        i = np.searchsorted(u_length[0], u)  # returns index i of the desired value u in u_length[0]
        # such that u_length[0][i-1] < u_desired <= u_length[0][i]
        # -> u_desired between u_length[0][i-1] and u_length[0][i]
        # linear interpolation
        length = u_length[1][i - 1] + (u - u_length[0][i - 1]) / (u_length[0][i] - u_length[0][i - 1]) * \
            (u_length[1][i] - u_length[1][i - 1])

    return length


def u_arc_to_u(u_arc, u_length):
    # transforms values of u_arc in [0, 1] to u in [0, 1], with u_arc being the fraction of the spline length based on
    # the data given by u_length - and u being the fraction of the spline length assuming line segments
    # between the knots
    if type(u_length) == float:
        return u_arc
    if type(u_arc) == list:
        u_arc = np.array(u_arc)
    length = u_arc * get_length(u_length)

    return length_to_u(length, u_length)


def u_to_u_arc(u, u_length):
    # transforms values of u in [0, 1] to u_arc in [0, 1], with u_arc being the fraction of the spline length based on
    # the data given by u_length - and u being the fraction of the spline length assuming line segments
    # between the knots
    if type(u_length) == float:
        return u
    if type(u) == list:
        u = np.array(u)
    length = u_to_length(u, u_length)
    u_arc = length / get_length(u_length)

    return u_arc


def u_arc_to_length(u_arc, u_length):
    return u_arc * get_length(u_length)


def length_to_u(length, u_length):
    # computes the corresponding length to an array of u_desired values in [0.0, 1.0]
    if type(length) == list:
        length = np.array(length)
    length = np.minimum(length, get_length(u_length))
    if type(u_length) in [int, float]:
        # assuming line segments between the knots, the relation is linear
        u = length / u_length
    else:
        i = np.searchsorted(u_length[1], length)  # returns index i of the desired length in u_length[1]
        # such that u_length[1][i-1] < length_desired <= u_length[1][i]
        # -> length_desired between u_length[1][i-1] and u_length[1][i]
        # linear interpolation
        u = u_length[0][i - 1] + (length - u_length[1][i - 1]) / (u_length[1][i] - u_length[1][i - 1]) * \
            (u_length[0][i] - u_length[0][i - 1])

    return u


def get_length(u_length):
    if type(u_length) in [int, float]:
        return u_length
    else:
        return u_length[1][-1]


def convert_joint_space_to_cartesian_space(env, joint_data):
    target_link_point_list = env._robot_scene.obstacle_wrapper._target_link_point_list

    if joint_data.ndim == 2:
        joint_data = joint_data.T
    else:
        joint_data = joint_data.reshape(1, len(joint_data))

    num_target_points = len(target_link_point_list)
    cartesian_data = [[] for _ in range(num_target_points)]  # [target_point][data_point][cartesian pos and orn]
    for i in range(len(joint_data)):
        target_position = np.zeros(env._robot_scene.num_manip_joints)
        target_position[env._robot_scene.spline_joint_mask] = joint_data[i]
        env._robot_scene.obstacle_wrapper.set_robot_position_in_obstacle_client(
            target_position=target_position)

        for j in range(num_target_points):
            pos, orn = target_link_point_list[j].get_position(
                return_orn=True)
            pos_orn_list = list(pos)
            pos_orn_list.extend(orn)
            cartesian_data[j].append(pos_orn_list)

    return np.array(cartesian_data)


def add_debug_line(line_from, line_to, debug_line_buffer, debug_line_buffer_index, line_color, line_width,
                   physics_client_id):
    if debug_line_buffer_index < len(debug_line_buffer[0]):
        dummy_id = p.addUserDebugLine(line_from,
                                      line_to,
                                      lineColorRGB=line_color, lineWidth=line_width,
                                      replaceItemUniqueId=debug_line_buffer[0][debug_line_buffer_index],
                                      physicsClientId=physics_client_id)
    else:
        debug_line_id = -1
        retry_attempts = 10
        busy_wait(time_in_micro_seconds=300)
        while debug_line_id == -1 and retry_attempts > 0:
            debug_line_id = p.addUserDebugLine(line_from,
                                               line_to,
                                               lineColorRGB=line_color, lineWidth=line_width,
                                               physicsClientId=physics_client_id)

            retry_attempts = retry_attempts - 1
        debug_line_buffer[0].append(debug_line_id)


def busy_wait(time_in_micro_seconds=1):
    current_time = time.perf_counter()
    target_time = current_time + time_in_micro_seconds * 1e-6
    while time.perf_counter() < target_time:
        pass

