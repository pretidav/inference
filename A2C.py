import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class Actor:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    def create_model(self):
        # Encoder
        input = tf.keras.layers.Input(shape=self.state_dim)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        # Decoder
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="linear", padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="linear", padding="same")(x)
        out = tf.keras.layers.Conv2D(1, (3, 3), activation="tanh", padding="same")(x)
        
        return tf.keras.models.Model(input, out)

    def get_action(self, state):
        state = np.reshape(state, [1,28,28])
        pert = self.model.predict(state)
        return pert[0]

    def get_noisy_action(self,state,time,alpha):
        return self.get_action(state) + alpha/np.sqrt(time) * np.random.uniform(size=(28,28,1),low=-1,high=1)

    def compute_loss(self, actions, pert, advantages):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(actions, pert, sample_weight=tf.stop_gradient(advantages))


    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            pert = self.model(states, training=True)
            loss = self.compute_loss( 
                                    actions=actions,
                                    pert=pert,
                                    advantages=advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    def create_model(self):
        state_input = tf.keras.layers.Input(shape=self.state_dim)
        x = tf.keras.layers.Conv2D(10, (3, 3), activation="relu", padding="same")(state_input)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Conv2D(10, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10, activation='linear')(x)
        v = tf.keras.layers.Dense(1, activation='linear')(x)

        return tf.keras.models.Model(state_input, v)

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    @tf.function
    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class A2CAgent:
    def __init__(self, data_shape,seed=1988):
        np.random.seed(seed=seed)
        self.state_dim = data_shape
        self.action_dim = data_shape 
        self.action_bound = 1
        self.gamma = 0.99
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound)
        self.critic = Critic(self.state_dim)

    def td_target(self, reward, next_state, done): 
        if done:
            return reward
        v_value = self.critic.model.predict(np.reshape(next_state,[1,28,28]))
        return np.reshape(reward + self.gamma * v_value[0], [1, 1])

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

