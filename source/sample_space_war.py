from pettingzoo.atari import space_war_v2
from DQN import DQN, ReplayBuffer
import supersuit
import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from collections import deque
from centroid_vectorization import detect_ship_centroids, build_feature_vector

env = space_war_v2.parallel_env(render_mode="human")
#env = supersuit.color_reduction_v0(env, mode='full')
env = supersuit.resize_v1(env, 84, 84)
env = supersuit.frame_stack_v1(env, 4)
env = supersuit.dtype_v0(env, np.float32)

observations, infos = env.reset()

for agent, ob in observations.items():
    print("OBs shape for", agent, ":", ob.shape)
    break

#setting up DQN agent
dqn_agent = env.agents[0]
observation_shape = env.observation_space(dqn_agent).shape
n_actions = env.action_space(dqn_agent).n
policy = DQN(observation_shape, n_actions)
target = DQN(observation_shape, n_actions)

LOAD_EPISODE = 0
if LOAD_EPISODE>0:
    policy.load_state_dict(torch.load(f"dqn_spacewar_ep{LOAD_EPISODE}.pth"))
    target.load_state_dict(torch.load(f"target_spacewar_ep{LOAD_EPISODE}.pth"))
    print(f"Loaded checkpoint ep{LOAD_EPISODE}")
else:
    target.load_state_dict(policy.state_dict())

optimizer = optim.Adam(policy.parameters(), lr=0.0001)
buffer = ReplayBuffer()

gamma = 0.99
epsilon = 1.0
batch_size = 32
update_target_every = 2000
step_count = 0
episode=LOAD_EPISODE
max_episodes=10+LOAD_EPISODE
while True:
    while env.agents:
        actions = {}
        for agent, ob in observations.items():

            if random.random() < epsilon:
                actions[agent] = env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    inp = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)
                    q = policy(inp)
                    actions[agent] = int(torch.argmax(q))

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        

        # save transitions per agent
        for agent in observations:

            obs = next_obs[agent]
            centroid_info = detect_ship_centroids(obs)
            feature_vec = build_feature_vector(agent, centroid_info)
            if feature_vec is not None:
                # Unpack relevant features
                relative_angle = feature_vec[7]
                distance = feature_vec[6]
                # Use relative angle for alignment reward
                alignment_reward = 0.02 * np.cos(relative_angle)
                # Encourage staying alive slightly
                survival_reward = 0.005
                # Encourage proximity (negative of distance)
                proximity_reward = -0.001 * distance
                fired = infos[agent].get("shot_fired", False)  # safely get 'shot_fired' flag
                shoot_reward = 0.1 if (fired and abs(relative_angle) < 0.26) else 0
                base_reward = rewards[agent]  # This is environment reward; you may keep or ignore

                # Combine partial rewards with base reward
                combined_reward = base_reward + alignment_reward + survival_reward + proximity_reward + shoot_reward

                # Save transition with vector input and combined reward
                buffer.add((
                    feature_vec.astype(np.float32),        # vectorized observation instead of raw pixels
                    actions[agent],
                    combined_reward,
                    build_feature_vector(agent, detect_ship_centroids(next_obs.get(agent, obs))).astype(np.float32),  # next state vector
                    terms[agent] or truncs[agent]
                ))
            else:
                print(f"Warning: Could not build feature vector for agent {agent}")
                exit()

            

        observations = next_obs
        step_count += 1

        # --- learn ---
        if len(buffer) > batch_size:
            s, a, r, s2, d = buffer.sample(batch_size)

            q_vals = policy(s).gather(1, a.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q = target(s2).max(1)[0]
                target_q = r + gamma * next_q * (1 - d)

            loss = ((q_vals - target_q) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- slowly reduce epsilon ---
        epsilon = max(0.05, epsilon * 0.9999)

        # --- periodic target update ---
        if step_count % update_target_every == 0:
            target.load_state_dict(policy.state_dict())
    episode += 1
    print(f"Episode {episode} completed.")
    if episode % 10 == 0:
        
        exit()
        torch.save(policy.state_dict(), f"dqn_spacewar_ep{episode}.pth")
        torch.save(target.state_dict(), f"target_spacewar_ep{episode}.pth")
        print(f"Saved checkpoint at episode {episode}")

    observations, _ = env.reset()
    if episode >= max_episodes:
        break
env.close()