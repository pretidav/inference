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

def take_step(state,target_image, action, pos): 
    image_flat = state
    image_flat[pos]=action
    next_state = np.reshape(image_flat,newshape=(28,28,1))    
    print(np.shape(next_state))
    print(np.shape(target_image))
    reward, done = get_reward(attack_image=next_state, target_image=target_image)
    return next_state, reward, done


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=1)
    parser.add_argument("--episodes", default=1)
    # parser.add_argument("--epoch", default=10)
    # parser.add_argument("--train", action='store_true')
    # parser.add_argument("--serve", action='store_true')
    # parser.add_argument("--predict", action='store_true')
    
    args = parser.parse_args()

    train_images, train_labels, test_images, test_labels, class_names = get_dataset()
    target_image = train_images[0]
    target_label = train_labels[0]

    state = np.ndarray.flatten(train_images[0])
    state_label = train_labels[0]

    
    agent = A2CAgent(data_shape=np.shape(state)[0])
    agent.actor.model.summary()
    agent.critic.model.summary()
    batch_size=args.batch
    num_of_episodes = args.episodes

    ep = 0 
    won = 0
    
    for i_episode in tqdm(range(num_of_episodes)):
        state_batch = []
        action_batch = []
        td_target_batch = []
        advantage_batch = []
        episode_reward, done = 0, False
        
        while True:
            action, pos = agent.actor.get_action(state)
            action = np.clip(action, 0, agent.action_bound)
            print(action,pos)
            next_state, reward, done = take_step(state=state, 
                                                    target_image=target_image, 
                                                    action=action, 
                                                    pos=pos)
            state = np.reshape(state, [1, agent.state_dim])
            action = np.reshape(action, [1, agent.action_dim])
            next_state = np.reshape(next_state, [1, agent.state_dim])
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
                td_targets = agent.list_to_batch(td_target_batch)
                advantages = agent.list_to_batch(advantage_batch)
                actor_loss = agent.actor.train(states, actions, advantages)
                critic_loss = agent.critic.train(states, td_targets)    
                state_batch = []
                action_batch = []
                td_target_batch = []
                advantage_batch = []

            episode_reward += reward[0][0]
            state = next_state[0]

            if done:
                won+=1
                print('won')
                print("Episode {} finished with {} mean reward".format(i_episode+1,episode_reward))
                break

        if won==50:
            print('--- won ---')
            agent.actor.model.save('./models/actor.hdf5',overwrite=True,include_optimizer=False)
            agent.critic.model.save('./models/critic.hdf5',overwrite=True,include_optimizer=False)
            break
            