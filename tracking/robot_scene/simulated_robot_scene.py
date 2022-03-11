# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import pybullet as p
import os
from PIL import Image, ImageDraw
from tracking.robot_scene.robot_scene_base import RobotSceneBase


class SimRobotScene(RobotSceneBase):
    COLOR_GREEN = [2 / 255, 138 / 255, 1 / 255, 1]
    COLOR_BLUE = [70 / 255, 100 / 255, 170 / 255, 1]
    COLOR_RED = [162 / 255, 34 / 255, 35 / 255, 1]
    SPHERE_COLOR = [COLOR_RED, COLOR_BLUE, COLOR_GREEN]  # default rgba color of the sphere
    BOARD_COLOR_DEFAULT = [0.85, 0.85, 0.85]  # alpha channel not supported right now
    BOARD_SIZE = (0.3393, 0.2715, 0.002)
    BOARD_BORDER_OFFSET = 0.03
    BOARD_SPHERE_POSITIONING_OFFSET = 0.06

    def __init__(self,
                 balancing_randomize_sphere=True,
                 balancing_random_initial_sphere_position=True,
                 balancing_noisy_sphere_position=False,
                 **kwargs):

        super().__init__(**kwargs)

        self._board_index_list = []
        self._board_color_edge = []

        if self._sphere_balancing_mode:
            self._board_index_list = self._get_board_index_list(link_name="iiwa_link_board")

            if self._visual_mode and self._simulation_client_id is not None:
                self._tex_uid_list = []
                for i in range(self._num_robots):
                    self._tex_uid_list.append(p.loadTexture(os.path.join(self.URDF_DIR, "board", "tex256.png")))
                    p.changeVisualShape(self._robot_id, self._board_index_list[i],
                                        textureUniqueId=self._tex_uid_list[i],
                                        physicsClientId=self._simulation_client_id)

            for i in range(len(self.SPHERE_COLOR)):
                self._board_color_edge.append(self.SPHERE_COLOR[i][:3])
                # rgb color of the board's edge, alpha channel not supported

        self._sphere_urdf = os.path.join(self.URDF_DIR, "sphere", "sphere.urdf")
        self._sphere_id_list = [None] * self._num_robots
        self._sphere_radius_list = [None] * self._num_robots
        self._sphere_start_pos_list = [None] * self._num_robots
        self._sphere_start_pos_rel_list = [None] * self._num_robots
        self._sphere_orn_list = [None] * self._num_robots

        self._balancing_randomize_sphere = balancing_randomize_sphere
        self._balancing_random_initial_sphere_position = balancing_random_initial_sphere_position
        self._balancing_noisy_sphere_position = balancing_noisy_sphere_position

        self._actual_base_pos = None
        self._actual_base_orn = None
        self._robot_z_angle_deviation_rad = None

    def pose_manipulator(self, joint_positions):
        p.resetJointStatesMultiDof(bodyUniqueId=self._robot_id,
                                   jointIndices=self._manip_joint_indices,
                                   targetValues=[[pos] for pos in joint_positions],
                                   targetVelocities=[[0.0] for _ in joint_positions],
                                   physicsClientId=self._simulation_client_id)

        if self._sphere_balancing_mode:
            self._place_sphere()

    def get_actual_joint_position_and_velocity(self, manip_joint_indices=None, physics_client_id=0):
        if manip_joint_indices is None:
            manip_joint_indices = self._manip_joint_indices
        # return the actual joint position and velocity for the specified joint indices from the physicsClient
        joint_states = p.getJointStates(self._robot_id, manip_joint_indices,
                                        physicsClientId=physics_client_id)

        joint_states_swap = np.swapaxes(np.array(joint_states, dtype=object), 0, 1)

        return joint_states_swap[0], joint_states_swap[1]

    @staticmethod
    def send_command_to_trajectory_controller(target_positions, **kwargs):
        pass

    def _get_board_index_list(self, link_name):
        board_index_list = []
        link_name_list = self.get_link_names_for_multiple_robots(link_name)
        for i in range(len(link_name_list)):
            board_index_list.append(self.get_link_index_from_link_name(link_name_list[i]))
        return board_index_list

    def _place_sphere(self, color_index=0):
        def adapt_board_visualization(robot=0, initial_sphere_position_rel_board=(0, 0)):
            if self._visual_mode:
                tex_array = np.array(self.BOARD_COLOR_DEFAULT * 256 * 256) * 255
                tex_array = tex_array.reshape(256, 256, 3)
                image = Image.fromarray(np.uint8(tex_array))
                draw = ImageDraw.Draw(image)
                edge_color = tuple(np.uint8(np.floor(np.array(
                    self._board_color_edge[color_index % len(self._board_color_edge)]) * 255)))
                draw.line([board_to_picture([-1, -1]), board_to_picture([-1, 1])], fill=edge_color, width=4)
                draw.line([board_to_picture([-1, 1]), board_to_picture([1, 1])], fill=edge_color, width=5)
                draw.line([board_to_picture([1, 1]), board_to_picture([1, -1])], fill=edge_color, width=4)
                draw.line([board_to_picture([1, -1]), board_to_picture([-1, -1])], fill=edge_color, width=5)
                pixel_x_2 = 1
                pixel_y_2 = 1
                initial_ball_position_pixel = board_to_picture(initial_sphere_position_rel_board)
                draw.ellipse((initial_ball_position_pixel[0] - pixel_x_2,
                              initial_ball_position_pixel[1] - pixel_y_2,
                              initial_ball_position_pixel[0] + pixel_x_2,
                              initial_ball_position_pixel[1] + pixel_y_2), fill=edge_color)

                tex_array = np.array(image)
                p.changeTexture(self._tex_uid_list[robot], list(tex_array.reshape(256 * 256 * 3)), 256, 256)

        def board_to_picture(rel_board_pos=(0.0, 0.0)):
            board_pixel = [63, 63]  # the board has 64 x 64 pixels
            pic_bottom_left = np.array([255 - board_pixel[0], 255 - board_pixel[1]])
            pic_bottom_right = np.array([255, 255 - board_pixel[1]])
            pic_top_left = np.array([255 - board_pixel[0], 255])

            rel_picture_pos = np.floor(pic_bottom_left +
                                       (pic_bottom_right - pic_bottom_left) * (rel_board_pos[0] + 1) / 2 +
                                       (pic_top_left - pic_bottom_left) * (rel_board_pos[1] + 1) / 2)

            return tuple(rel_picture_pos)

        for j in range(self._num_robots):
            if self._sphere_id_list[j] is not None:
                for i in range(self._num_clients):
                    p.removeBody(self._sphere_id_list[j], physicsClientId=i)

            sphere_properties = self._get_sphere_properties()
            self._sphere_radius_list[j] = sphere_properties[0]
            sphere_pos_global, sphere_orn_global, sphere_pos_local_rel_board = \
                self._get_init_sphere_pose(robot=j)

            for i in range(self._num_clients):
                if i == 0:
                    base_position = sphere_pos_global
                else:
                    base_position = [0, 0, -1.0]
                self._sphere_id_list[j] = p.loadURDF(self._sphere_urdf, basePosition=base_position,
                                                     baseOrientation=sphere_orn_global,
                                                     globalScaling=sphere_properties[0],
                                                     useFixedBase=False,
                                                     physicsClientId=i)
            p.changeDynamics(self._sphere_id_list[j], -1, mass=sphere_properties[1],
                             rollingFriction=sphere_properties[2], spinningFriction=sphere_properties[3])

            p.changeVisualShape(self._sphere_id_list[j], -1,
                                rgbaColor=self.SPHERE_COLOR[color_index % len(self.SPHERE_COLOR)])

            adapt_board_visualization(robot=j, initial_sphere_position_rel_board=sphere_pos_local_rel_board)

            self._sphere_start_pos_list[j], self._sphere_start_pos_rel_list[j] = \
                self.get_sphere_position_on_board(robot=j)

    def _get_sphere_properties(self):
        # (radius[m], mass[kg], rolling friction coefficient, spinning friction coefficient)
        if self._balancing_randomize_sphere:
            sphere_characteristics = [np.random.randint(20, 37) / 1000,
                                      np.random.randint(2, 204) / 1000,
                                      np.random.randint(1, 7) / 10000,
                                      np.random.randint(1, 7) / 10000]
        else:
            sphere_characteristics = [27.5 / 1000,
                                      102 / 1000,
                                      3 / 10000,
                                      3 / 10000]

        return sphere_characteristics

    def _get_init_sphere_pose(self, robot=0):
        def get_randomized_init_sphere_position():
            placement_field = list(0.5 * np.array(self.BOARD_SIZE) - self.BOARD_BORDER_OFFSET -
                                   self.BOARD_SPHERE_POSITIONING_OFFSET)

            x_offset = np.random.uniform(-placement_field[0], placement_field[0])
            y_offset = np.random.uniform(-placement_field[1], placement_field[1])
            sphere_pos = [x_offset, y_offset, self.BOARD_SIZE[2]]
            x_rel = x_offset / (0.5 * self.BOARD_SIZE[0])
            y_rel = y_offset / (0.5 * self.BOARD_SIZE[1])
            sphere_pos_rel = [x_rel, y_rel]

            return sphere_pos, sphere_pos_rel

        sphere_orn_local = [0, 0, 0, 1]
        if self._balancing_random_initial_sphere_position:
            sphere_pos_global, sphere_pos_local_rel_board = get_randomized_init_sphere_position()
        else:
            sphere_pos_global = [0, 0, self.BOARD_SIZE[2]]
            sphere_pos_local_rel_board = [0, 0]
        # sphere_pos_local_rel_board is relative to the board without considering BOARD_BORDER_OFFSET
        sphere_pos_global[2] += self._sphere_radius_list[robot]
        # now transform the ball pose into the world coordinate system
        board_pos, board_orn = self._get_board_pose(robot)
        sphere_pos_global, sphere_orn_global = map(list, p.multiplyTransforms(board_pos, board_orn,
                                                                              sphere_pos_global, sphere_orn_local))

        return sphere_pos_global, sphere_orn_global, sphere_pos_local_rel_board

    def _get_board_pose(self, robot=0):
        board_pose = p.getLinkState(self._robot_id, self._board_index_list[robot], computeForwardKinematics=True,
                                    physicsClientId=self._simulation_client_id)
        board_pos, board_orn = list(board_pose[0]), list(board_pose[1])
        board_pos[2] -= 0.5 * self.BOARD_SIZE[2]

        return board_pos, board_orn

    def get_sphere_position_on_board(self, robot=0):
        def apply_noise(sphere_pos):
            if self._balancing_noisy_sphere_position:
                noise_range = 0.002
                sphere_pos[0] += np.random.uniform(-noise_range, noise_range)
                sphere_pos[1] += np.random.uniform(-noise_range, noise_range)

            return sphere_pos

        sphere_pos_global = p.getBasePositionAndOrientation(self._sphere_id_list[robot],
                                                            physicsClientId=self._simulation_client_id)[0]
        sphere_pos_local = self._compute_local_sphere_position(sphere_pos_global, robot)
        sphere_pos_local = apply_noise(sphere_pos_local)

        sphere_pos_local_rel = list(np.array(sphere_pos_local[:2]) /
                                    (0.5 * np.array(self.BOARD_SIZE[:2]) - self.BOARD_BORDER_OFFSET))
        sphere_pos_local_rel.append(sphere_pos_local[2])

        return sphere_pos_local, sphere_pos_local_rel

    def _compute_local_sphere_position(self, sphere_position_global, robot=0):
        board_pose = self._get_board_pose(robot)
        # transform the current ball position into local coordinates of the board
        board_pos_inv, board_orn_inv = p.invertTransform(board_pose[0], board_pose[1])
        sphere_position_local, _ = map(list, p.multiplyTransforms(board_pos_inv, board_orn_inv,
                                                                  sphere_position_global, [0, 0, 0, 1]))
        sphere_position_local[2] -= self._sphere_radius_list[robot]

        return list(sphere_position_local)

    def check_if_robot_lost_balance(self, z_angle_deviation_deg_max=30):
        robot_lost_balance = False
        if self.robot_z_angle_deviation_rad / np.pi * 180 > z_angle_deviation_deg_max:
            robot_lost_balance = True

        return robot_lost_balance

    def clear_last_action(self):
        super().clear_last_action()
        self._actual_base_pos = None
        self._actual_base_orn = None
        self._robot_z_angle_deviation_rad = None

    def convert_point_from_static_base_to_floating_base(self, pos_static, orn_static):
        pos_floating, orn_floating = p.multiplyTransforms(positionA=self.actual_base_pos,
                                                          orientationA=self.actual_base_orn,
                                                          positionB=pos_static,
                                                          orientationB=orn_static)
        return pos_floating, orn_floating

    @property
    def robot_z_angle_deviation_rad(self):
        if self._robot_z_angle_deviation_rad is None:
            # compute angle deviation from the z-axis
            z_axis_transformed, _ = p.multiplyTransforms(positionA=[0, 0, 0],
                                                         orientationA=self.actual_base_orn,
                                                         positionB=[0, 0, 1],
                                                         orientationB=(0, 0, 0, 1))

            if z_axis_transformed[0] == 0 and z_axis_transformed[1] == 0:
                self._robot_z_angle_deviation_rad = 0
            else:
                self._robot_z_angle_deviation_rad = \
                    0.5 * np.pi - np.arctan(z_axis_transformed[2] / np.linalg.norm(z_axis_transformed[0:2]))

        return self._robot_z_angle_deviation_rad

    @property
    def robot_base_pos_deviation(self):
        return np.linalg.norm(self.actual_base_pos)

    @property
    def robot_base_orn_deviation(self):
        # there are various metrics to compute the "distance" between quaternions
        # we use distance = arccos(abs(<q_1, q_2>)) * 2 / pi with <> being the dot product
        # This leads to a normalized distance range [0, 1] with 0 meaning no distance and 1 meaning the maximum distance
        # the reference orientation is (0, 0, 0, 1) which results in a simpler equation in our case
        # distance = arccos(abs(q_1[3])) * 2 / pi
        return np.arccos(abs(self.actual_base_orn[3])) * 2 / np.pi

    @property
    def sphere_start_pos_list(self):
        return self._sphere_start_pos_list

    @property
    def sphere_start_pos_rel_list(self):
        return self._sphere_start_pos_rel_list

    @property
    def actual_base_pos(self):
        if self._actual_base_pos is None:
            actual_base_pos, actual_base_orn = \
                p.getBasePositionAndOrientation(self._robot_id, self._simulation_client_id)
            self._actual_base_pos = np.array(actual_base_pos)
            self._actual_base_orn = np.array(actual_base_orn)
        return self._actual_base_pos

    @property
    def actual_base_orn(self):
        if self._actual_base_orn is None:
            actual_base_pos, actual_base_orn = \
                p.getBasePositionAndOrientation(self._robot_id, self._simulation_client_id)
            self._actual_base_pos = np.array(actual_base_pos)
            self._actual_base_orn = np.array(actual_base_orn)
        return self._actual_base_orn






