# Code  adapted from https://github.com/ray-project/ray/blob/master/rllib/models/tf/fcnet.py
# Copyright 2021 Ray Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


class FullyConnectedNetworkLastLayerActivation(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, last_layer_activation):
        super(FullyConnectedNetworkLastLayerActivation, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        activation = get_activation_fn(model_config.get("fcnet_activation"))
        if last_layer_activation is not None:
            last_layer_activation = get_activation_fn(last_layer_activation)
        hiddens = model_config.get("fcnet_hiddens")
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")

        # we are using obs_flat, so take the flattened shape as input
        inputs = tf.keras.layers.Input(
            shape=(np.product(obs_space.shape), ), name="observations")
        last_layer = inputs
        i = 1

        if no_final_linear:
            # the last layer is adjusted to be of size num_outputs
            for size in hiddens[:-1]:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)
                i += 1
            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
        else:
            # the last layer is a linear if last_layer_activation is None, else last_layer_activation
            for size in hiddens:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)
                i += 1
            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=last_layer_activation,
                kernel_initializer=normc_initializer(0.01))(last_layer)

        if not vf_share_layers:
            # build a parallel set of hidden layers for the value net
            last_layer = inputs
            i = 1
            for size in hiddens:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)
                i += 1

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(last_layer)

        self.base_model = tf.keras.Model(inputs, [layer_out, value_out])
        # self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
