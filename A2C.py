import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    def create_model(self):
        
        def rescale(a):
            return a*tf.constant(self.action_bound,dtype=tf.float32)
        
        def rescale_pos(a):
            return a*tf.constant(self.state_dim-1,dtype=tf.float32)
                #tf.cast(a*tf.constant(self.state_dim-1,dtype=tf.float32),dtype=tf.int32)
        

        state_input = tf.keras.layers.Input((self.state_dim,))
        dense_1 = tf.keras.layers.Dense(50, activation='linear')(state_input)
        dense_2 = tf.keras.layers.Dense(32, activation='linear')(dense_1)
        
        out_mu = tf.keras.layers.Dense(self.action_dim, activation='sigmoid')(dense_2)   
        std_output = tf.keras.layers.Dense(self.action_dim, activation='softplus')(dense_2) 
        mu_output = out_mu #tf.keras.layers.Lambda(lambda x: rescale(x))(out_mu)
        
        dense_3  = tf.keras.layers.Dense(16, activation='linear')(dense_2)
        position   = tf.keras.layers.Dense(1, activation='sigmoid')(dense_3)   
        position_output = tf.keras.layers.Lambda(lambda x: rescale_pos(x))(position)
        
        return tf.keras.models.Model(state_input, [mu_output, std_output, position_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std, pos = self.model.predict(state)
        mu, std, pos = mu[0], std[0], pos[0]
        
        return np.random.normal(mu, std, size=self.action_dim), pos

    def compute_loss(self, mu, std, actions, pos, pos_pred, advantages):
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        mse_loss = tf.keras.losses.MeanSquaredError() 
        loss_policy = (-dist.log_prob(value=actions) * advantages + 0.001*dist.entropy())
        return tf.reduce_sum(loss_policy) + mse_loss(tf.cast(pos_pred,tf.float32),tf.cast(pos,tf.float32))
        
    def train(self, states, actions, pos, advantages):
        with tf.GradientTape() as tape:
            mu, std, pos_pred = self.model(states, training=True)
            loss = self.compute_loss(mu=mu, 
                                    std=std, 
                                    actions=actions,
                                    pos_pred=pos_pred,
                                    pos=pos,
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
        state_input = tf.keras.layers.Input((self.state_dim,))
        dense_1 = tf.keras.layers.Dense(50, activation='relu')(state_input)
        dense_2 = tf.keras.layers.Dense(32, activation='relu')(dense_1)
        dense_3 = tf.keras.layers.Dense(16, activation='relu')(dense_2)
        v       = tf.keras.layers.Dense(1, activation='linear')(dense_3)
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
    def __init__(self, data_shape):
        self.state_dim = data_shape
        self.action_dim = 1 
        self.action_bound = 1
        self.std_bound = [1e-3, 1.0]
        self.gamma = 0.99
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model.predict(
            np.reshape(next_state, [1, self.state_dim]))
        return np.reshape(reward + self.gamma * v_value[0], [1, 1])

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

