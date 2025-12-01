from pettingzoo.atari import space_war_v2
from DQN import DQN, ReplayBuffer
import supersuit

env = space_war_v2.parallel_env(render_mode="human")
env = supersuit.color_reduction_v0(env, mode='full')
env = supersuit.resize_v1(env, 100, 100)
env = supersuit.frame_stack_v1(env, 4)
observations, infos = env.reset()

#setting up DQN agent
dqn_agent = env.agents[0]
observation_shape = env.observation_space(dqn_agent).shape
n_actions = env.action_space(dqn_agent).n
policy = DQN(obs_shape, n_actions)
target = DQN(obs_shape, n_actions)
target.load_state_dict(policy.state_dict())
optimizer = optim.Adam(policy.parameters(), lr=0.0001)
buffer = ReplayBuffer()

gamma = 0.99
epsilon = 1.0
batch_size = 32
update_target_every = 2000
step_count = 0


while env.agents:
    actions = {}
    for agent, ob in obs.items():

        if random.random() < epsilon:
            actions[agent] = env.action_space(agent).sample()
        else:
            with torch.no_grad():
                inp = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)
                q = policy(inp)
                actions[agent] = int(torch.argmax(q))

    next_obs, rewards, terms, truncs, infos = env.step(actions)

    # save transitions per agent
    for agent in obs:
        buffer.add((
            obs[agent],
            actions[agent],
            rewards[agent],
            next_obs.get(agent, obs[agent]),   # handle agent death
            terms[agent] or truncs[agent]
        ))

    obs = next_obs
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

    # --- reset episode ---
    if not env.agents:
        obs, _ = env.reset()
env.close()