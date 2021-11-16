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
    
    reward = - (np.max(attack_pred)-np.min(attack_pred)) 
    done = False 
    if reward > 9.0: 
        done = True
    return reward, done 

def take_step(state, target_image, action, eps, counter): 
    next_state = state + eps*action    
    next_state = np.clip(next_state, 0, 1)
    reward, done = get_reward(attack_image=next_state, target_image=target_image)
    if counter>1000: 
        done = True
    return next_state, reward, done


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=1)
    parser.add_argument("--episodes", default=1)
    parser.add_argument("--eps", default=0.001)
    
    args = parser.parse_args()

    train_images, train_labels, test_images, test_labels, class_names = get_dataset()
    target_image = train_images[0]
    target_label = train_labels[0]

    state = train_images[0]
    state_label = train_labels[0]


    agent = A2CAgent(data_shape=state.shape)
    agent.actor.model.summary()
    agent.critic.model.summary()
    batch_size=int(args.batch)
    num_of_episodes = int(args.episodes)
    eps = float(args.eps)

    ep = 0 
    won = 0
    for i_episode in tqdm(range(num_of_episodes)):
        count = 0
        
        target_image = train_images[0]
        target_label = train_labels[0]
        state        = train_images[0]
        state_label  = train_labels[0]
    
        reward_history  = []
        state_batch     = []
        action_batch    = []
        pos_batch       = []
        td_target_batch = []
        advantage_batch = []
        episode_reward, done = 0, False
        
        while True:
            count +=1
            #action = agent.actor.get_action(state)
            action = agent.actor.get_noisy_action(state=state, time=count, alpha=0.01)
            action = np.clip(action, 0, agent.action_bound)
            next_state, reward, done = take_step(state=state, 
                                                    target_image=target_image, 
                                                    action=action, 
                                                    eps=eps,
                                                    counter=count)
            if count%100==0:
                print('epoch {}: reward {}'.format(count,reward))
            reward_history.append(reward)
            
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
                break

        # if won==50:
        #     print('--- won ---')
        #     agent.actor.model.save('./models/actor.hdf5',overwrite=True,include_optimizer=False)
        #     agent.critic.model.save('./models/critic.hdf5',overwrite=True,include_optimizer=False)
        #     break
            