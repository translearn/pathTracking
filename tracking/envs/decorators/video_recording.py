# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import time
import pybullet as p
import json
import numpy as np
from glob import glob
from abc import ABC
from PIL import Image, ImageDraw, ImageFont, ImageGrab
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tracking.envs.tracking_base import TrackingBase

RENDER_MODES = ["human", "rgb_array"]
VIDEO_FRAME_RATE = 60
ASPECT_RATIO = 16/9
VIDEO_HEIGHT = 1080


def compute_projection_matrix():
    fov = 90
    near_distance = 0.1
    far_distance = 100

    return p.computeProjectionMatrixFOV(fov, ASPECT_RATIO, near_distance, far_distance)


class VideoRecordingManager(ABC, TrackingBase):
    # Renderer
    OPENGL_GUI_RENDERER = 0
    OPENGL_EGL_RENDERER = 1
    CPU_TINY_RENDERER = 2
    IMAGEGRAB_RENDERER = 3

    def __init__(self,
                 *vargs,
                 extra_render_modes=None,
                 video_frame_rate=None,
                 video_height=VIDEO_HEIGHT,
                 video_dir=None,
                 fixed_video_filename=False,
                 camera_angle=0,
                 render_video=False,
                 renderer=OPENGL_GUI_RENDERER,
                 render_no_shadows=False,
                 **kwargs):

        # video recording settings
        if video_frame_rate is None:
            video_frame_rate = VIDEO_FRAME_RATE
        if video_height is None:
            video_height = VIDEO_HEIGHT

        self._video_height = video_height
        self._video_width = int(ASPECT_RATIO * self._video_height)
        self._render_video = render_video
        self._renderer = renderer

        super().__init__(*vargs, **kwargs)

        self._video_recorder = None
        if video_dir is None:
            self._video_dir = self._evaluation_dir
        else:
            self._video_dir = video_dir
        self._fixed_video_filename = fixed_video_filename
        self._video_base_path = None
        self._render_modes = RENDER_MODES.copy()
        if extra_render_modes:
            self._render_modes += extra_render_modes
        self._sim_steps_per_frame = int(1 / (video_frame_rate * self._control_time_step))
        self._video_frame_rate = 1 / (self._sim_steps_per_frame * self._control_time_step)

        self._camera_angle = camera_angle
        self._cam_dis, self._cam_yaw, self._cam_pitch, self._cam_target_pos, self._cam_roll = \
            self.get_camera_angle_settings(self._camera_angle)
        self._view_matrix = self.compute_view_matrix()
        self._projection_matrix = compute_projection_matrix()
        self._sim_step_counter = None

        if self._gui_client_id is not None:
            gpu_support = 'CUDA' in os.environ['PATH'].upper() if 'PATH' in os.environ else False
            use_shadows = 1 if not render_no_shadows else 0
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, use_shadows, lightPosition=[-15, 0, 28],
                                       shadowMapResolution=16384 if gpu_support else 4096,
                                       physicsClientId=self._gui_client_id)
            # set shadowMapResolution to 4096 when running the code without a dedicated GPU and to 16384 otherwise

    def get_camera_angle_settings(self, camera_angle=0):
        if camera_angle == 0:
            cam_target_pos = (0, 0, 0)
            cam_dis = 1.75
            cam_yaw = 90
            cam_pitch = -70
            cam_roll = 0

        elif camera_angle == 1:
            cam_target_pos = (-0.25, 0, 0)
            cam_dis = 1.95
            cam_yaw = 90
            cam_pitch = -40
            cam_roll = 0

        elif camera_angle == 2:
            cam_yaw = 59.59992599487305
            cam_pitch = -49.400054931640625
            cam_dis = 2.000002861022949
            cam_target_pos = (0.0, 0.0, 0.0)
            cam_roll = 0

        elif camera_angle == 3:
            cam_yaw = 64.39994049072266
            cam_pitch = -37.000003814697266
            cam_dis = 2.000002861022949
            cam_target_pos = (0.0, 0.0, 0.0)
            cam_roll = 0

        elif camera_angle == 4:
            cam_yaw = 69.59991455078125
            cam_pitch = -33.8000602722168
            cam_dis = 1.8000028133392334
            cam_target_pos = (0.0, 0.0, 0.0)
            cam_roll = 0

        elif camera_angle == 5:
            cam_yaw = 90.800048828125
            cam_pitch = -59.800079345703125
            cam_dis = 1.8000028133392334
            cam_target_pos = (0.0, 0.0, 0.0)
            cam_roll = 0

        elif camera_angle == 6:
            cam_yaw = 90.4000473022461
            cam_pitch = -65.40008544921875
            cam_dis = 2.000002861022949
            cam_target_pos = (0.0, 0.0, 0.0)
            cam_roll = 0

        elif camera_angle == 7:
            cam_yaw = 90.00004577636719
            cam_pitch = -45.4000358581543
            cam_dis = 2.000002861022949
            cam_target_pos = (0.0, 0.0, 0.0)
            cam_roll = 0

        elif camera_angle == 8:
            cam_yaw = 89.60002899169922
            cam_pitch = -17.400007247924805
            cam_dis = 1.4000000953674316
            cam_target_pos = (-0.07712450623512268, 0.05323473736643791, 0.45070940256118774)
            cam_roll = 0

        elif camera_angle == 9:
            cam_yaw = 90  # 91.20002746582031
            cam_pitch = -29.0
            cam_dis = 2.0000
            cam_target_pos = (0.0, -0.04, 0.58)
            cam_roll = 0

        elif camera_angle == 10:
            cam_yaw = 90
            cam_pitch = -28.999988555908203
            cam_dis = 2.3000
            cam_target_pos = (0.000, 0, 0.5800000429153442)
            cam_roll = 0

        elif camera_angle == 11:
            cam_yaw = 90
            cam_pitch = -28.999988555908203
            cam_dis = 1.5
            cam_target_pos = (0.000, 0, 0.5800000429153442)
            cam_roll = 0

        elif camera_angle == 12:
            cam_yaw = 90
            cam_pitch = 0
            cam_dis = 1.5
            cam_target_pos = (0.000, 0, 0.5800000429153442)
            cam_roll = 0

        elif camera_angle == 13:
            cam_target_pos = (0, 0, 0)
            cam_dis = 1.75
            cam_yaw = 90
            cam_pitch = -89
            cam_roll = 0

        else:
            raise ValueError("camera_angle {} is not defined".format(camera_angle))

        return cam_dis, cam_yaw, cam_pitch, cam_target_pos, cam_roll

    def compute_view_matrix(self):
        return p.computeViewMatrixFromYawPitchRoll(self._cam_target_pos, self._cam_dis, self._cam_yaw, self._cam_pitch,
                                                   self._cam_roll, 2)

    def reset(self, **kwargs):
        super().reset(**kwargs)

        self._sim_step_counter = 0
        if self._render_video:
            if self._renderer == self.IMAGEGRAB_RENDERER:
                time.sleep(0.05)
            self._reset_video_recorder()

    def close(self):
        if self._video_recorder:
            self._close_video_recorder()

        super().close()

    def _sim_step(self):
        super()._sim_step()
        self._sim_step_counter += 1

        if self._render_video:
            if self._sim_step_counter == self._sim_steps_per_frame:
                self._capture_frame_with_video_recorder()

    def _prepare_for_end_of_episode(self):
        super()._prepare_for_end_of_episode()

        if self._render_video:
            self._capture_frame_with_video_recorder(frames=int(self._video_frame_rate))
            if self._video_recorder:
                self._close_video_recorder()

    def _capture_frame_with_video_recorder(self, frames=1):
        self._sim_step_counter = 0
        self._video_recorder.capture_frame()
        for _ in range(frames - 1):
            self._video_recorder._encode_image_frame(self._video_recorder.last_frame)

    @property
    def metadata(self):
        metadata = {
            'render.modes': self._render_modes,
            'video.frames_per_second': self._video_frame_rate
        }

        return metadata

    def _reset_video_recorder(self):
        if self._video_recorder:
            self._close_video_recorder()

        os.makedirs(self._video_dir, exist_ok=True)

        episode_id = self._episode_counter
        if self._fixed_video_filename:
            self._video_base_path = os.path.join(self._video_dir, "episode_{}".format(episode_id))
        else:
            self._video_base_path = os.path.join(self._video_dir, "episode_{}_{}".format(episode_id, self.pid))

        metadata = {
            'video.frames_per_second': self._video_frame_rate,
            'video.height': self._video_height,
            'video.width': self._video_width,
            'video.camera_angle': self._camera_angle,
            'episode_id': episode_id,
            'seed': self._fixed_seed
        }

        self._video_recorder = VideoRecorder(self, base_path=self._video_base_path, metadata=metadata, enabled=True)

        self._capture_frame_with_video_recorder(frames=int(self._video_frame_rate))

    def render(self, mode="human"):
        if mode == "human":
            return np.array([])
        else:
            physics_client_id = self._simulation_client_id if not self._switch_gui else self._obstacle_client_id
            if self._renderer != self.IMAGEGRAB_RENDERER:
                (_, _, image, _, _) = p.getCameraImage(width=self._video_width, height=self._video_height,
                                                       renderer=self._pybullet_renderer,
                                                       viewMatrix=self._view_matrix,
                                                       projectionMatrix=self._projection_matrix,
                                                       shadow=0, lightDirection=[-20, -0.5, 150],
                                                       flags=p.ER_NO_SEGMENTATION_MASK,
                                                       physicsClientId=physics_client_id)
                image = np.reshape(image, (self._video_height, self._video_width, 4))
                image = np.uint8(image[:, :, :3])
            else:
                p.resetDebugVisualizerCamera(cameraDistance=self._cam_dis, cameraYaw=self._cam_yaw,
                                             cameraPitch=self._cam_pitch,
                                             cameraTargetPosition=self._cam_target_pos,
                                             physicsClientId=physics_client_id)
                if self._episode_counter == 1 and self._episode_length == 0:
                    time.sleep(0.15)
                image = ImageGrab.grab()

            return np.array(image)

    def _close_video_recorder(self):
        self._video_recorder.close()
        self._video_recorder = []
        if not self._fixed_video_filename:
            self._adapt_rendering_metadata()
            self._rename_output_files()

    def _adapt_rendering_metadata(self):
        metadata_ext = ".meta.json"
        metadata_file = self._video_base_path + metadata_ext

        with open(metadata_file, 'r') as f:
            metadata_json = json.load(f)

            encoder_metadata = metadata_json.pop('encoder_version', None)
            if encoder_metadata:
                metadata_json.update(encoder=encoder_metadata['backend'])

            metadata_json.update(trajectory_length=self._trajectory_manager.trajectory_length)
            metadata_json.update(episode_length=self._episode_length)
            metadata_json.update(total_reward=round(self._total_reward, 3))

        with open(metadata_file, 'w') as f:
            f.write(json.dumps(metadata_json, indent=4))
            f.close()

    def _rename_output_files(self):
        output_file = glob(self._video_base_path + ".*")

        for file in output_file:
            dir_path, file_name = os.path.split(file)
            name, extension = os.path.splitext(file_name)
            new_name = "_".join(map(str, [name, self._episode_length, round(self._total_reward, 3)]))
            if self._use_target_points:
                new_name = "_".join(map(str, [new_name] +
                                        [self._robot_scene.obstacle_wrapper.get_num_target_points_reached(robot=robot)
                                         for robot in range(self._robot_scene.num_robots)]
                                        + [self._robot_scene.obstacle_wrapper.get_num_target_points_reached()]))
            if self._use_splines:
                spline_name, _ = os.path.splitext(self._trajectory_manager.spline_name)
                new_name = "_".join(map(str, [new_name, spline_name, self._termination_reason]))

            new_file_name = new_name + extension
            os.rename(file, os.path.join(dir_path, new_file_name))
