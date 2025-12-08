import torch
import random
from SpaceWarGame import SpaceEnv
from DQN import DQN
from DQN import ReplayBuffer
import copy

def load_model(path,input_size,n_actions):
    model = DQN(input_size, n_actions)
    try:
        model.load_state_dict(torch.load(path))
        print("Loaded:", path)
    except:
        print("No saved model found:",path)
    return model
number_episodes_trained=0
#follows all steps of training -the modifying
def play(player1,player2):
    env=SpaceEnv(render=True)
    if random.random() < 0.5:
        env.add_agent("green")
        env.add_agent("blue")
    else:
        env.add_agent("blue")
        env.add_agent("green")
    env.stats["green"]= {"wins":0, "losses":0, "games":0, "winrate":0.0}
    env.stats["blue"]= {"wins":0, "losses":0, "games":0, "winrate":0.0}
    input_size=33
    n_actions=5
    green_policy = load_model(player1, input_size, n_actions)
    blue_policy = load_model(player2, input_size, n_actions)
    win_history=[]
    recent_window = 200
    #no randomization
    epsilon=0
    env.render_enabled=False
    for episode in range(1000):
        #render every 100th game
        """if episode%100==0:
            env.render_enabled=True
        else:
            env.render_enabled=False"""
        env.reset()
        if random.random() < 0.5:
            env.add_agent("green")
            env.add_agent("blue")
        else:
            env.add_agent("blue")
            env.add_agent("green")

        done = False
        play_count=0
        while not done:
            if play_count>1000:
                break
            play_count+=1
            obs = {name: env.build_obs(name) for name in env.agents}

            # agent actions
            actions = {}

            for name, observation in obs.items():
                model = green_policy if name == "green" else blue_policy
                qvals = model(torch.tensor(observation).float().unsqueeze(0))
                actions[name] = int(torch.argmax(qvals))

            # step env
            next_obs, rewards, dones = env.step(actions)

            done = any(dones.values())
        winner, loser = env.record_outcome()
        win_history.append(winner)
        if len(win_history) > recent_window:
            win_history.pop(0)
        if episode%10==0:
            print(f"Models trained {number_episodes_trained}: {(episode/10):.2f}%", end='\r')
    
    g_wr = env.stats["green"]["winrate"]
    b_wr = env.stats["blue"]["winrate"]
    print(f"With {number_episodes_trained} episodes of training, MADRL WR: {g_wr}, Vanilla DQN WR: {b_wr}")
    #print("Green WR:", g_wr)
    #print("Blue  WR:", b_wr)
    #print(f"Games played - Green: {env.stats['green']['games']}, Blue: {env.stats['blue']['games']}")

if __name__ == "__main__":
    for sets in range(0,40):
        play(f"trained_models/madrl_bullets_{number_episodes_trained}.pth",f"trained_models/dqn_bullets_{number_episodes_trained}.pth")
        number_episodes_trained+=500