from numpy.core.shape_base import block
from A2C import A2CAgent
import numpy as np
import matplotlib.pyplot as plt 
import random
from tqdm import tqdm
from train_serve_predict import get_dataset
import argparse
import json
import requests

def get_prediction(attack_image, target_image):
    image = np.array([attack_image,target_image])
    data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/model:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    attack_prediction, target_prediction = predictions
    return attack_prediction, target_prediction

def get_reward(attack_image,target_image):
    attack_pred, target_pred = get_prediction(attack_image=attack_image, target_image=target_image)
    gold_class   = np.argmax(target_pred)
    attack_class = np.argmax(attack_pred)
    #print(target_pred)
    #print(attack_pred)
    reward =  (attack_pred[1]) 
    done = False 
    if reward > 0.9: 
        done = True
    return reward, done 

def take_step(state, target_image, action, eps, counter): 
    next_state = state + eps*action    
    next_state = np.clip(next_state, 0, 1)
    reward, done = get_reward(attack_image=next_state, target_image=target_image)
    if counter>200: 
        done = True
    return next_state, reward, done



class Display():
    def __init__(self, time): 
        print('## Display ##')
        self.time = time
        self.start()

    def start(self):
        plt.ion()
        plt.show() 
        plt.figure(figsize=(20, 8))
        ax0 = plt.subplot(2, 1, 2)
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)

    def end(self):
        plt.show()

    def display(self,img1, img2, time, episode, reward):
        """
        Displays 2 images from each one of the supplied arrays.
        """
        plt.suptitle('episode: {} epoch: {} reward: {}'.format(episode,time,reward[-1]))
        
        plt.subplot(2, 1, 2)
        plt.title('reward')
        plt.plot([i for i in range(0,len(reward))], reward, 'r-')
        plt.ylabel('reward')
        plt.xlabel('t')

        plt.subplot(2, 2, 1)
        plt.title('initial state')
        plt.imshow(img1.reshape(28, 28))
        plt.gray()

        plt.subplot(2, 2, 2)
        plt.title('current state')
        plt.imshow(img2.reshape(28, 28))
        plt.gray()

        plt.draw()
        plt.pause(self.time)

def random_image(shape): 
    return np.random.rand(*shape)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=8)
    parser.add_argument("--episodes", default=10)
    parser.add_argument("--eps", default=0.1)
    parser.add_argument("--alpha", default=1)
    parser.add_argument("--randomstart", action='store_true')

    args = parser.parse_args()
    disp = Display(time=5)

    train_images, train_labels, test_images, test_labels, class_names = get_dataset()
    
    agent = A2CAgent(data_shape=train_images[0].shape)
    agent.actor.model.summary()
    agent.critic.model.summary()
    batch_size=int(args.batch)
    num_of_episodes = int(args.episodes)
    eps = float(args.eps)
    alpha = float(args.alpha)

    ep = 0 
    won = 0

    if args.randomstart: 
        target_image = random_image(shape=train_images[0].shape)
        target_label = 0 
    else :     
        target_image = train_images[0]
        target_label = train_labels[0]
    
    for i_episode in tqdm(range(num_of_episodes)):
        count = 0

        
        state = target_image
        state_label = target_label

    
        reward_history  = []
        state_batch     = []
        action_batch    = []
        pos_batch       = []
        td_target_batch = []
        advantage_batch = []
        episode_reward, done = 0, False
        
        disp.display(img1=target_image, img2=state, time=0, episode=i_episode, reward='-')
        
        while True:
            count +=1
            action = agent.actor.get_noisy_action(state=state, time=count, alpha=alpha)
            action = np.clip(action, 0, agent.action_bound)
            next_state, reward, done = take_step(state=state, 
                                                    target_image=target_image, 
                                                    action=action, 
                                                    eps=eps,
                                                    counter=count)
            reward_history.append(reward)
            
            if count%50==0:
                print('epoch {}: reward {}'.format(count,reward))
                disp.display(img1=target_image, img2=next_state, time=count, episode=i_episode, reward=reward_history)

            
            action = np.reshape(action,  [1, 28, 28])
            next_state = np.reshape(next_state,  [1, 28, 28])
            state = np.reshape(state,  [1, 28, 28])
            

            
            reward = np.reshape(reward, [1, 1])
            td_target = agent.td_target(reward, next_state, done)
            advantage = agent.advantage(td_target, agent.critic.model.predict(state))
           
            state_batch.append(state)
            action_batch.append(action)
            td_target_batch.append(td_target)
            advantage_batch.append(advantage)
            
            if len(state_batch) >= batch_size or done:    
                states = agent.list_to_batch(state_batch)
                actions = agent.list_to_batch(action_batch)
                actions=np.expand_dims(actions,axis=3)
                td_targets = agent.list_to_batch(td_target_batch)
                advantages = agent.list_to_batch(advantage_batch)
               
                actor_loss = agent.actor.train(states=states, 
                                                actions=actions, 
                                                advantages=advantages)
                critic_loss = agent.critic.train(states=states, 
                                                td_targets=td_targets)    
                state_batch     = []
                action_batch    = []
                td_target_batch = []
                advantage_batch = []

            episode_reward += reward[0][0]
            state = np.reshape(next_state[0],[28,28,1])
            
            if episode_reward/count>0.8:
                x = [i for i in range(count)]
                plt.scatter(x=x,y=reward_history)
                plt.xlabel('epochs')
                plt.legend(['reward'])
                plt.show()
                #break

            if done:
                won+=1
                print('won')
                print("Episode {} finished with {} mean reward".format(i_episode+1,episode_reward/count))
                disp.end()
                break




    x = [i for i in range(count)]
    plt.scatter(x=x,y=reward_history)
    plt.xlabel('epochs')
    plt.legend(['reward'])
    plt.show()

        # if won==50:
        #     print('--- won ---')
        #     agent.actor.model.save('./models/actor.hdf5',overwrite=True,include_optimizer=False)
        #     agent.critic.model.save('./models/critic.hdf5',overwrite=True,include_optimizer=False)
        #     break
            