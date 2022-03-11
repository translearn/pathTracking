# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
from klimits import normalize_batch as normalize_batch


class SplinePlotter:
    def __init__(self,
                 u_arc_interpolation_step=0.00001,
                 normalize=False,
                 pos_limits=None,
                 der_1_limits=None,
                 der_2_limits=None,
                 der_3_limits=None,
                 cartesian_limits=None,
                 plot_dimension=None,
                 plot_norm=False):

        self._u_arc_interpolation_step = u_arc_interpolation_step
        self._normalize = normalize
        self._plot_dimension = plot_dimension
        self._plot_norm = plot_norm
        self._cartesian_limits = np.asarray(cartesian_limits)

        self._pos_limits = np.asarray(pos_limits).T if pos_limits is not None else None
        self._der_1_limits = np.asarray(der_1_limits).T if der_1_limits is not None else None
        self._der_2_limits = np.asarray(der_2_limits).T if der_2_limits is not None else None
        self._der_3_limits = np.asarray(der_3_limits).T if der_3_limits is not None else None

        self._episode_counter = 0
        self._joint_splines = None
        self._cartesian_splines = None
        self._fig = None
        self._ax = None
        self._ax_indices = None

        # you can select the backend for matplotlib with matplotlib.use(backend_str)
        # potential values for backend_str with GUI support: 'QT5Agg', 'TkAgg'
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['text.usetex'] = False

    def reset_plotter(self):
        self._episode_counter = self._episode_counter + 1
        self._joint_splines = []
        self._cartesian_splines = []

    def display_plot(self, spline_type="joint", x_lim=None, blocking=True):

        num_subplots = 4

        self._fig, self._ax = plt.subplots(num_subplots, 1, sharex=True)

        plt.subplots_adjust(left=0.05, bottom=0.04, right=0.95, top=0.98, wspace=0.15, hspace=0.15)

        ax_pos = 0
        ax_offset = 1
        ax_der_1 = 0 + ax_offset
        ax_der_2 = 1 + ax_offset
        ax_der_3 = 2 + ax_offset

        self._ax_indices = [ax_pos, ax_der_1, ax_der_2, ax_der_3]

        if spline_type == "joint":
            self._plot_joint_spline(x_lim)
        else:
            self._plot_cartesian_spline(x_lim)

        self._fig.set_size_inches((24.1, 13.5), forward=False)
        if blocking:
            logging.info("Trajectory plotted. Close plot to continue")
        plt.show(block=blocking)

    def _clear_axes(self):
        for i in range(len(self._ax)):
            self._ax[i].clear()

    def _plot_joint_spline(self, xlim=None, clear_axes=True):
        if clear_axes:
            self._clear_axes()
        fig = self._fig
        ax = self._ax
        ax_pos = self._ax_indices[0]
        ax_der_1 = self._ax_indices[1]
        ax_der_2 = self._ax_indices[2]
        ax_der_3 = self._ax_indices[3]

        for i in range(len(ax)):
            ax[i].grid(True)
        ax[-1].set_xlabel('Spline length')

        if ax_pos is not None:
            ax[ax_pos].set_ylabel('Position')

        if ax_der_1 is not None:
            ax[ax_der_1].set_ylabel('First Derivative')

        if ax_der_2 is not None:
            ax[ax_der_2].set_ylabel('Second Derivative')

        if ax_der_3 is not None:
            ax[ax_der_3].set_ylabel('Third Derivative')

        max_length = 0

        for i in range(len(self._joint_splines)):
            spline = self._joint_splines[i]
            marker = spline['marker']
            u_arc_interpolation = spline['object'].get_interpolation(sample_distance=self._u_arc_interpolation_step,
                                                                     use_normalized_sample_distance=True,
                                                                     u_start=spline['object'].u_start,
                                                                     u_end=spline['object'].u[
                                                                         spline['object'].u_end_index])

            interpolated_length = u_arc_interpolation * spline['object'].get_length()
            start_length = interpolated_length[0]
            interpolated_length = interpolated_length - start_length  # relative to length at u_start
            interpolated_length_diff = interpolated_length[1] - interpolated_length[0]
            # interpolated_length_diff is constant when sampling from u_arc

            relevant_knot_indices = np.logical_and(spline['object'].u_start <= spline['object'].u,
                                                   spline['object'].u <= spline['object'].u[
                                                                         spline['object'].u_end_index])
            knot_length = spline['object'].u_to_length(u=spline['object'].u[relevant_knot_indices])
            knot_length = knot_length - start_length
            max_length = max(max_length, interpolated_length[-1])
            interpolated_pos = spline['object'].evaluate_u_arc(u_arc=u_arc_interpolation)
            knot_pos = spline['object'].curve_data_spline[:, relevant_knot_indices]
            interpolated_der_1 = np.diff(interpolated_pos, axis=1) / interpolated_length_diff
            interpolated_der_2 = np.diff(interpolated_der_1, axis=1) / interpolated_length_diff
            interpolated_der_3 = np.diff(interpolated_der_2, axis=1) / interpolated_length_diff

            if self._normalize and self._pos_limits is not None:
                interpolated_pos_plot = normalize_batch(interpolated_pos.T, self._pos_limits).T
                knot_pos_plot = normalize_batch(knot_pos.T, self._pos_limits).T
            else:
                interpolated_pos_plot = interpolated_pos
                knot_pos_plot = knot_pos

            if self._normalize and self._der_1_limits is not None:
                interpolated_der_1_plot = normalize_batch(interpolated_der_1.T, self._der_1_limits).T
            else:
                interpolated_der_1_plot = interpolated_der_1

            if self._normalize and self._der_2_limits is not None:
                interpolated_der_2_plot = normalize_batch(interpolated_der_2.T, self._der_2_limits).T
            else:
                interpolated_der_2_plot = interpolated_der_2

            if self._normalize and self._der_3_limits is not None:
                interpolated_der_3_plot = normalize_batch(interpolated_der_3.T, self._der_3_limits).T
            else:
                interpolated_der_3_plot = interpolated_der_3

            for j in range(len(interpolated_pos)):
                if self._plot_dimension is None or self._plot_dimension[j]:
                    color = 'C' + str(j)  # "C0", "C1" -> index to the default color cycle
                    if spline['label'] != '_nolegend_':
                        label = "Spline " + str(i + 1) if spline['label'] is None else spline['label']
                        label = label + ": Joint " + str(j + 1)
                    else:
                        label = '_nolegend_'
                    if ax_pos is not None:
                        ax[ax_pos].plot(knot_length, knot_pos_plot[j], color=color,
                                        marker=marker, linestyle='None', label='_nolegend_')
                        ax[ax_pos].plot(interpolated_length, interpolated_pos_plot[j],
                                        color=color, linestyle=spline['linestyle'], label=label)
                    if ax_der_1 is not None:
                        ax[ax_der_1].plot(interpolated_length[:-1], interpolated_der_1_plot[j],
                                          color=color, linestyle=spline['linestyle'], label=label)

                    if ax_der_2 is not None:
                        ax[ax_der_2].plot(interpolated_length[1:-1], interpolated_der_2_plot[j],
                                          color=color, linestyle=spline['linestyle'], label=label)

                    if ax_der_3 is not None:
                        ax[ax_der_3].plot(interpolated_length[1:-2], interpolated_der_3_plot[j],
                                          color=color, linestyle=spline['linestyle'], label=label)

            if self._plot_norm:
                color = 'black'
                if spline['label'] != '_nolegend_':
                    label = "Spline " + str(i + 1) if spline['label'] is None else spline['label']
                    label = label + ": Norm"
                else:
                    label = '_nolegend_'
                if ax_pos is not None:
                    interpolated_pos_norm = np.linalg.norm(interpolated_pos_plot, axis=0)
                    knot_pos_norm = np.linalg.norm(knot_pos_plot, axis=0)
                    ax[ax_pos].plot(knot_length, knot_pos_norm, color=color,
                                    marker=marker, linestyle='None', label='_nolegend_')
                    ax[ax_pos].plot(interpolated_length, interpolated_pos_norm,
                                    color=color, linestyle=spline['linestyle'], label=label)
                if ax_der_1 is not None:
                    interpolated_der_1_norm = np.linalg.norm(interpolated_der_1_plot, axis=0)
                    ax[ax_der_1].plot(interpolated_length[:-1], interpolated_der_1_norm,
                                      color=color, linestyle=spline['linestyle'], label=label)

                if ax_der_2 is not None:
                    interpolated_der_2_norm = np.linalg.norm(interpolated_der_2_plot, axis=0)
                    ax[ax_der_2].plot(interpolated_length[1:-1], interpolated_der_2_norm,
                                      color=color, linestyle=spline['linestyle'], label=label)

                    logging.info("Maximum curvature {} at {:.3f}".format(np.max(interpolated_der_2_norm),
                                                                         interpolated_length[np.argmax(
                                                                             interpolated_der_2_norm) + 1]))

                    plot_integrated_curvature = True
                    if plot_integrated_curvature:
                        interpolated_der_2_norm_integration = np.concatenate(([0.0],
                                                                              np.cumsum(interpolated_der_2_norm) *
                                                                              interpolated_length_diff))
                        ax[ax_der_2].plot(interpolated_length[0:-1], interpolated_der_2_norm_integration,
                                          color="grey", linestyle=spline['linestyle'], label=label)

                if ax_der_3 is not None:
                    interpolated_der_3_norm = np.linalg.norm(interpolated_der_3_plot, axis=0)
                    ax[ax_der_3].plot(interpolated_length[1:-2], interpolated_der_3_norm,
                                      color=color, linestyle=spline['linestyle'], label=label)

        ax[-1].legend(loc='lower right')

        for i in range(len(ax)):
            if xlim is None:
                ax[i].set_xlim([0, max_length])
            else:
                ax[i].set_xlim(xlim)

        if self._normalize:
            if ax_pos is not None and self._pos_limits is not None and not self._plot_norm:
                ax[ax_pos].set_ylim([-1.05, 1.05])

        fig.align_ylabels(ax)

    def _plot_cartesian_spline(self, xlim=None, clear_axes=True):
        pass

    def add_joint_spline(self, spline, label=None, linestyle="-", marker="."):
        self._joint_splines.append({'object': spline, 'label': label, 'linestyle': linestyle,
                                    'marker': marker})


def normalize(value, value_range):
    normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
    continuous_joint_indices = np.isnan(value_range[0]) | np.isnan(value_range[1])
    if np.any(continuous_joint_indices):
        # continuous joint -> map [-np.pi, np.pi] and all values shifted by 2 * np.pi to [-1, 1]
        if np.array(value).ndim == 2:  # batch computation
            normalized_value[:, continuous_joint_indices] = \
                -1 + 2 * (((value[:, continuous_joint_indices] + np.pi) / (2 * np.pi)) % 1)
        else:
            normalized_value[continuous_joint_indices] = \
                -1 + 2 * (((value[continuous_joint_indices] + np.pi)/(2 * np.pi)) % 1)
    return normalized_value


def denormalize(norm_value, value_range):
    if np.isnan(value_range[0]) or np.isnan(value_range[1]):  # continuous joint
        value_range[0] = -np.pi
        value_range[1] = np.pi
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
