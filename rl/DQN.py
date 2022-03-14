# this is going to be the DQN itself
# understand that this will be using concepts regarding double DQN
# essentially, we are training one model and copying over the weights to the other model
# libraries to be imported
from matplotlib.pyplot import plot
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


# GLOBAL VARIABLES
learning_rate = 0.0001
BATCH_SIZE = 128


# the class for the DQN
class DQN:

    # constructor
    def __init__(self, num_states, num_actions, model_path):

        # printing some of the values for debugging later
        print('num_states:', num_states)
        print('num_actions:', num_actions)

        # defining some data fields
        self.num_states = num_states
        self.num_actions = num_actions

        # Base model
        self.model = self.build_model()

        # Target model (essentially a copy of the base model itself)
        self.model_ = self.build_model()

        # creating model checkpoints and callback functions
        self.model_checkpoint_1 = model_path + "CarRacing_DDQN_model_best.h5"
        self.model_checkpoint_2 = model_path + "CarRacing_DDQN_model_per.h5"

        save_best = ModelCheckpoint(
            self.model_checkpoint_1,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min',
            period=20
        )
        save_per = ModelCheckpoint(
            self.model_checkpoint_2,
            monitor='loss',
            verbose=0,
            save_best_only=False,
            mode='min',
            period=400
        )

        self.callbacks_list = [save_best, save_per]


    # CNN that takes the state and outputs the Q values for all of the possible actions
    # Model
    #     - in : states
    #     - out : actions
    def build_model(self):

        states_in = Input(shape=self.num_states, name='states_in')
        x = Conv2D(32, (8,8), strides=(4,4), activation='relu', name='conv1')(states_in)
        x = Conv2D(64, (4,4), strides=(2,2), activation='relu', name='conv2')(x)
        x = Conv2D(64, (3,3), strides=(1,1), activation='relu', name='conv3')(x)
        x = Flatten(name='flattened')(x)
        x = Dense(512, activation='relu')(x)
        out = Dense(self.num_actions, activation='linear')(x)

        model = Model(states_in, out)
        self.optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False
        )
        loss = losses.MeanSquaredError()
        model.compile(
            loss=loss,
            optimizer=self.optimizer
        )
        plot_model(
            model,
            to_file='model_architecture.png',
            show_shapes=True
        )

        return model


    # training the function itself
    def train(self, x, y, epochs=10, verbose=0):
        self.model.fit(
            x,
            y,
            batch_size=(BATCH_SIZE),
            epochs=epochs,
            verbose=verbose,
            callbacks=self.callbacks_list,
        )


    # predict function
    def predict(self, state, target=False):
        if target:
            # return the Q value given a state from the target network
            return self.model_.predict(state)
        else:
            # return the Q value from the original network
            return self.model.predict(state)


    # predicting only a single state
    def predict_single_state(self, state, target=False):
        x = state[np.newaxis, :, :, :]
        return self.predict(x, target)


    # update the target model with the base model weights
    def target_model_update(self):
        self.model_.set_weights(self.model.get_weights())

