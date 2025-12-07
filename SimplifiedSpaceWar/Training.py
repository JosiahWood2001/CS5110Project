import torch
import random
from SpaceWarGame import SpaceEnv
from DQN import DQN
from DQN import ReplayBuffer
import copy

#will store best models for evolutionary MADRL
elite_pool = []
MAX_ELITES = 10

#attempt to load file input
def load_model(path,input_size,n_actions):
    model = DQN(input_size, n_actions)
    try:
        model.load_state_dict(torch.load(path))
        print("Loaded:", path)
    except:
        print("No saved model found:",path)
    return model

#this will remove the losing color and replace it with the best of the elite models
def evolutionary_update(env, green_policy, blue_policy, elite_pool, recent_winrate):
    stats = env.stats
    #get the win rates for each agent
    print("\n=== Evolution Check ===")
    g_wr = stats["green"]["winrate"]
    b_wr = stats["blue"]["winrate"]
    print("Green WR:", g_wr)
    print("Blue  WR:", b_wr)
    print(f"Games played - Green: {stats['green']['games']}, Blue: {stats['blue']['games']}")
    if stats["green"]["games"] > 30 and stats["blue"]["games"] > 30:
        #based on winner, name loser and winner
        if g_wr < b_wr:
            loser_name = "green"
            loser_model = green_policy
            winner_model = blue_policy
            winner_wr = recent_winrate["blue"]
        else:
            loser_name = "blue"
            loser_model = blue_policy
            winner_model = green_policy
            winner_wr = recent_winrate["green"]
    else:
        print("Not enough games played — skipping evolution.\n")
        return green_policy, blue_policy
    #add winner to elite pool
    elite_pool.append(copy.deepcopy(winner_model))
    elite_pool = elite_pool[-MAX_ELITES:]
    print("Replacing:", loser_name)
    #check for max win rate from pool
    candidates = [winner_model] + elite_pool
    parent = max(candidates, key=lambda m: winner_wr)
    #create new mutated agent from the best
    new_agent = parent.clone_with_mutation(noise_scale=0.002)
    if loser_name == "green":
        green_policy = new_agent
    else:
        blue_policy = new_agent
    #send updated policy and the losers name for updating the buffer
    print(f"Replaced {loser_name} with a mutated elite model!\n")
    return green_policy, blue_policy, loser_name

def train(MA_train, elite_pool,render=False,load_files=["",""]):
    env=SpaceEnv(render=render)
    #randomize what order they're added so they don't always start on the same side
    if random.random() < 0.5:
        env.add_agent("green")
        env.add_agent("blue")
    else:
        env.add_agent("blue")
        env.add_agent("green")
    #prepare stats for tracking agent data
    env.stats["green"]= {"wins":0, "losses":0, "games":0, "winrate":0.0}
    env.stats["blue"]= {"wins":0, "losses":0, "games":0, "winrate":0.0}
    #
    #These define the input/output dimensions
    #
    #input shape is 
    """
    distance to enemy x, y
    velocity x, y, rotational
    enemy velocity x, y, rotational
    distnace to enemy, cosine of angle to enemy, sine of angle to enemy,
    cosine of angle of enemy to self, sine of angle of enemy to self
    for each of 5 bullets, position x, y, vx, vy
    """
    input_size=33
    #actions possible
    """
    nothin, rotatecw, rotateccw, accelerate forward, fire
    """
    n_actions=5
    #load in files for each policy or start with default random
    green_policy = load_model(load_files[0], input_size, n_actions)
    blue_policy = load_model(load_files[1], input_size, n_actions)

    #allow setting age of policies for saving accurate file names
    green_age = 0
    blue_age = 0

    #define targets and optimizers for training
    target_green = DQN(input_size, n_actions)
    target_blue = DQN(input_size, n_actions)
    target_green.load_state_dict(green_policy.state_dict())
    target_blue.load_state_dict(blue_policy.state_dict())
    opt_g = torch.optim.Adam(green_policy.parameters(), lr=1e-4)
    opt_b = torch.optim.Adam(blue_policy.parameters(),  lr=1e-4)
    #buffers will store game states for reference and training
    #(allow for consequences to be understood past a single frame)
    buffer_g = ReplayBuffer()
    buffer_b = ReplayBuffer()
    #define exploration parameters
    gamma = 0.99
    epsilon = 1.0
    training_steps = 0
    win_history=[]
    recent_window = 200

    #the number of games to be played
    for episode in range(20000):
        #every 1000 steps, add current agents to elite pool
        if episode%1000==0:
            if MA_train:
                elite_pool.append(copy.deepcopy(blue_policy))
                elite_pool.append(copy.deepcopy(green_policy))
                elite_pool = elite_pool[-MAX_ELITES:]

        #play out/display every 200th game
        if episode%200==0:
            env.render_enabled=True
        else:
            env.render_enabled=False
        #reset the environment for next game
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
            #add time out, at a time, the models had fired all bullets and simply avoided them indefinitely
            if play_count>1000:
                break
            play_count+=1
            #create observation frame for decision making
            obs = {name: env.build_obs(name) for name in env.agents}

            # agent actions
            actions = {}

            for name, observation in obs.items():
                #take a random action occasionally
                if random.random() < epsilon:
                    actions[name] = random.randint(0, n_actions - 1)
                else:
                    #call policy then use policy to get an action
                    model = green_policy if name == "green" else blue_policy
                    qvals = model(torch.tensor(observation).float().unsqueeze(0))
                    #discrete game so only 1 action
                    actions[name] = int(torch.argmax(qvals))

            # step env
            next_obs, rewards, dones = env.step(actions, MA_train)

            # store transitions into respective buffers
            for name in env.agents:
                done_flag = 1.0 if dones[name] else 0.0
                if name=="green":
                    buffer_g.add((
                        obs[name], 
                        actions[name], 
                        rewards[name], 
                        next_obs[name], 
                        done_flag
                    ))
                else:
                    buffer_b.add((
                        obs[name], 
                        actions[name], 
                        rewards[name], 
                        next_obs[name], 
                        done_flag
                    ))

            done = any(dones.values())

            # train if buffer ready
            ags=[]
            if MA_train:
                # Alternate training: only 1 agent trains per episode
                if episode%2==0:
                    ags.append((green_policy, target_green, opt_g, buffer_g))
                else:
                    ags.append((blue_policy, target_blue, opt_b, buffer_b))
            else:
                # Vanilla: train both every episode
                ags.append((green_policy, target_green, opt_g, buffer_g))
                ags.append((blue_policy, target_blue, opt_b, buffer_b))
            for model, target, opt, buffer in ags:
                if len(buffer) > 64:
                    #observations, actions,rewards,next_observations,done flag
                    s, a, r, s2, d = buffer.sample(64)
                    q = model(s).gather(1, a.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        max_next = target(s2).max(1)[0]
                        #define targets from possible maximizations
                        target_q = r + gamma * max_next * (1 - d)
                    #calculate losses
                    loss = ((q - target_q) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    model.record_loss(loss)
                training_steps += 1
                #every 500 steps report losses for training monitoring
                if training_steps % 500 == 0:
                    avg_loss_green = sum(green_policy.loss_history[-100:]) / 100
                    avg_loss_blue = sum(blue_policy.loss_history[-100:]) / 100
                    print(f"Training Step {training_steps}: Avg Loss Green: {avg_loss_green:.4f}, Avg Loss Blue: {avg_loss_blue:.4f}")
        
        #after game get results
        winner, loser = env.record_outcome()
        win_history.append(winner)
        if len(win_history) > recent_window:
            win_history.pop(0)

        recent_winrate = {
            "green": win_history.count("green") / len(win_history),
            "blue":  win_history.count("blue")  / len(win_history)
        }
        #every 400 games apply evolutionary theory to mutate an elite to replace losing agent
        if episode % 400 == 0 and episode > 0:
            if MA_train:
                green_policy, blue_policy, replaced_name = evolutionary_update(
                    env, green_policy, blue_policy, elite_pool, recent_winrate
                )
                if replaced_name=="green":
                    buffer_g=copy.deepcopy(buffer_b)
                else:
                    buffer_b=copy.deepcopy(buffer_g)
        #as the agents improve, decrease randomizaion
        epsilon = max(0.05, epsilon * 0.999)
        #store the 2 models every 500 games whether madrl model or vanill dqn
        if episode % 500 == 0:
            if MA_train:
                torch.save(green_policy.state_dict(), f"trained_models/madrl_bullets_{green_age+episode}.pth")
                torch.save(blue_policy.state_dict(),  f"trained_models/madrl2_bullets_{blue_age+episode}.pth")
            else:
                torch.save(green_policy.state_dict(), f"trained_models/dqn_bullets_{green_age+episode}.pth")
                torch.save(blue_policy.state_dict(),  f"trained_models/dqn2_bullets_{blue_age+episode}.pth")
            print("Saved checkpoints at episode", episode)

        if episode % 10 == 0:
            print("Episode", episode, "completed.")


if __name__ == "__main__":
    train(False,elite_pool,render=True,load_files=[
        "",
        ""])