# Learning Time-optimized Path Tracking with or without Sensory Feedback
[![IROS 2022](https://img.shields.io/badge/IROS-2022-%3C%3E)](https://iros2022.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2203.01968-B31B1B)](http://arxiv.org/abs/2203.01968)
[![GitHub issues](https://img.shields.io/github/issues/translearn/pathtracking)](https://github.com/translearn/safemotions/issues/) <br>

This repository contains the code and the neural networks used for our paper ["Learning Time-optimized Path Tracking with or without Sensory Feedback"](https://arxiv.org/abs/2203.01968).

![tracking_picture](https://user-images.githubusercontent.com/51738372/157730389-788b0b9b-81d9-43b1-88d5-eb54133983fb.png)

## Installation

The code is written in python and does not need to be compiled.
Simply clone the repository with

    git clone https://github.com/translearn/pathTracking.git

The required dependencies can be installed by running:

    pip install -r requirements.txt


## Pretrained networks

We provide pretrained networks for the robot systems shown in the figure above. 
To track paths from a random dataset with an industrial robot run


    python tracking/evaluate.py --use_gui --checkpoint=industrial/no_balancing/random  

Other networks can be executed by adjusting the checkpoint argument. 
All available networks are listed below:

<table width="100%">
    <thead>
        <tr>
            <th>Robot system</th>
            <th>Configuration</th>
            <th>Dataset</th>
            <th>Checkpoint</th>
        </tr>
    </thead>
    <tbody>
         <tr>
            <td rowspan=3>Kuka iiwa </td>
            <td rowspan=3>no additional objectives</td>
            <td>random</td>
            <td>--checkpoint=industrial/no_balancing/random</td>
        </tr>
        <tr>
            <td>target point</td>
            <td>--checkpoint=industrial/no_balancing/target_point </td>
        </tr>
        <tr>
            <td>ball balancing</td>
            <td>--checkpoint=industrial/no_balancing/ball_balancing </td>
        </tr>
        <tr>
           <td rowspan=2> Kuka with balance board </td>
           <td>no balancing reward</td>
           <td rowspan=2>ball balancing</td>
           <td>--checkpoint=industrial/balancing/no_balancing_reward</td>
        </tr>
        <tr>
            <td>balancing reward</td>
            <td>--checkpoint=industrial/balancing/balancing_reward</td>
        </tr>
        <tr>
           <td>ARMAR-6</td>
           <td>no additional objectives</td>
           <td>random</td>
           <td>--checkpoint=humanoid/armar6/random</td>
        </tr>
         <tr>
            <td rowspan=5>ARMAR-4 </td>
            <td rowspan=2>no additional objectives, fixed base and legs</td>
            <td>random</td>
            <td>--checkpoint=humanoid/armar4/no_balancing/random</td>
        </tr>
        <tr>
            <td>target point</td>
            <td>--checkpoint=humanoid/armar4/no_balancing/target_point</td>
        </tr>
        <tr>
            <td>no balancing reward, fixed legs</td>
            <td rowspan=3>target point</td>
            <td>--checkpoint=humanoid/armar4/balancing/no_balancing_reward</td>
        </tr>
        <tr>
            <td>balancing reward, fixed legs</td>
            <td>--checkpoint=humanoid/armar4/balancing/balancing_reward_fixed_legs</td>
        </tr>
        <tr>
            <td>balancing reward, controlled legs</td>
            <td>--checkpoint=humanoid/armar4/balancing/balancing_reward_controlled_legs</td>
        </tr>
    </tbody>
</table>


## Training

Networks can also be trained from scratch. For instance, path tracking with an industrial robot can be learned by running  
```bash
python tracking/train.py --logdir=tracking_training --name=industrial_no_balancing_random --robot_scene=0 --online_trajectory_time_step=0.1 --hidden_layer_activation=swish --online_trajectory_time_step=0.1 --online_trajectory_duration=16.0 --obstacle_scene=0 --target_link_offset="[0, 0, 0.126]" --last_layer_activation=tanh --no_log_std_activation --use_controller_target_velocities --spline_dir=industrial/random/train --spline_u_arc_start_range="[0.0, 0.8]" --spline_u_arc_diff_min=0.2 --spline_normalize_duration --spline_termination_max_deviation=0.25 --obs_spline_n_next=7 --obs_spline_add_length --obs_spline_add_distance_per_knot --spline_distance_max_reward=2.0 --spline_deviation_max_threshold=0.25 --punish_spline_max_deviation --spline_max_deviation_max_punishment=0.9 --punish_spline_mean_deviation --spline_mean_deviation_max_punishment=0.9  --spline_deviation_weighting_factors="[1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6]"  --batch_size_factor=6.0 --spline_braking_extra_time_steps=0 --terminate_on_robot_stop --solver_iterations=50 --iterations_per_checkpoint=50  --time=500
```

## Publication
The corresponding publication is available at [https://arxiv.org/abs/2203.01968](https://arxiv.org/abs/2203.01968).

[![Video](https://user-images.githubusercontent.com/51738372/157881536-77e4dd68-71ed-4074-bec7-88925e946455.png)](https://youtu.be/gCPN8mqPVHg)


## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.