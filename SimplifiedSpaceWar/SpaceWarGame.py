import numpy as np
import math
import pygame
import time

#each agent will have a ship
class Ship:
    def __init__(self,x,y,angle=0.0):
        self.x = x
        self.y = y
        self.angle = angle
        #velocities x, y and rotational
        self.vx = 0.0
        self.vy = 0.0
        self.vr = 0.0
        self.cooldown = 0
    def state(self):
        return np.array([self.x, self.y, self.angle, self.vx, self.vy], dtype=np.float32)

#bullets will be added and agents will not be hurt by their own
class Bullet:
    def __init__(self, x, y, vx, vy, owner):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.owner = owner

class SpaceEnv:
    def __init__(self, render=False):
        self.world_size = 600
        self.render_enabled=render
        self.fps=20
        self.bullet_speed=6
        self.ship_accel=0.4
        self.turn_accel=0.1
        self.max_bullets=5
        self.stats = {}
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((self.world_size, self.world_size))
            pygame.display.set_caption("SpaceWar")
        self.reset()
    def add_agent(self, name, policy=None):
        #ensure each agent is on a different side of the screen randomly positioned
        if len(self.agents) == 0:
            x = np.random.uniform(100, self.world_size/2 - 50)
        else:
            x = np.random.uniform(self.world_size/2 + 50, self.world_size - 100)
        self.agents[name] = {
            "ship":Ship(x,y=np.random.uniform(100, self.world_size - 100),
                angle=np.random.uniform(0, 2 * math.pi)),
            "policy": policy,
            "reward": 0.0,
            "alive": True,
        }
    """
    #This function will update games, wins, and losses counts for evolutionary improvement
    """
    def record_outcome(self):
        alive = [name for name, a in self.agents.items() if a["alive"]]
        if len(alive) == 1:
            winner = alive[0]
            loser  = [x for x in self.agents if x != winner][0]
            s = self.stats

            s[winner]["wins"] += 1
            s[winner]["games"] += 1
            s[winner]["winrate"] = s[winner]["wins"] / s[winner]["games"]

            s[loser]["losses"] += 1
            s[loser]["games"] += 1
            s[loser]["winrate"] = s[loser]["wins"] / s[loser]["games"]

            return winner, loser
        return None, None
    def reset(self):
        #clear all the components to the game
        self.bullets = []
        self.agents={}
        self.step_count = 0
        return True
    #This will build the observation vector that the agent will use
    def build_obs(self, agent_name):
        me = self.agents[agent_name]["ship"]
        #extendable potentially
        others = [name for name in self.agents if name != agent_name]
        enemy_name = others[0]
        enemy = self.agents[enemy_name]["ship"]
        #values will be normalized relative to myself
        dx = enemy.x - me.x
        dy = enemy.y - me.y
        dvx = enemy.vx - me.vx
        dvy = enemy.vy - me.vy
        distance = math.sqrt(dx**2+dy**2)
        #normalized to -pi to pi
        angle_to_enemy=((math.atan2(dy,dx) - me.angle)+math.pi)%(2*math.pi)-math.pi
        angle_from_enemy=((math.atan2(-dy,-dx) - enemy.angle)+math.pi)%(2*math.pi)-math.pi
        #build primary observations
        obs = np.array([
            dx/self.world_size,dy/self.world_size,
            me.vx,me.vy,me.vr,
            enemy.vx,enemy.vy,enemy.vr,
            distance,math.cos(angle_to_enemy),math.sin(angle_to_enemy),
            math.cos(angle_from_enemy),math.sin(angle_from_enemy),
        ], dtype=np.float32)
        #filter enemy bullets for observation
        enemy_bullets = [b for b in self.bullets if b.owner != agent_name]
    
        # Sort bullets by distance to me
        def bullet_distance(b):
            dx_b = b.x - me.x
            dy_b = b.y - me.y
            # Consider wrap-around distances to handle toroidal world
            dx_b = (dx_b + self.world_size/2) % self.world_size - self.world_size/2
            dy_b = (dy_b + self.world_size/2) % self.world_size - self.world_size/2
            return math.sqrt(dx_b*dx_b + dy_b*dy_b)
        
        enemy_bullets.sort(key=bullet_distance)
        
        max_bullets_to_include = 5
        default_val = [10.0, 10.0, 0.0, 0.0]  # very far away position normalized, zero velocity
        #for each bullet, in order of closeness put it into the observations
        for i in range(max_bullets_to_include):
            if i < len(enemy_bullets):
                b = enemy_bullets[i]
                dx_b = b.x - me.x
                dy_b = b.y - me.y
                # wrap-around correction
                dx_b = (dx_b + self.world_size/2) % self.world_size - self.world_size/2
                dy_b = (dy_b + self.world_size/2) % self.world_size - self.world_size/2
                
                dvx_b = b.vx - me.vx
                dvy_b = b.vy - me.vy
                
                # Normalize position relative to world size, velocity can be left as is or normalized by bullet_speed if desired
                obs=np.append(obs,[
                    dx_b / self.world_size,
                    dy_b / self.world_size,
                    dvx_b / self.bullet_speed,  # normalize velocity by bullet speed
                    dvy_b / self.bullet_speed,
                ])
            else:
                obs=np.append(obs,default_val)
        return obs
    #apply physics and check for collisions
    def step(self,actions,madrl=True):
        for name, agent in self.agents.items():
            ship = agent["ship"]
            #skip all agents currently dead
            if not agent["alive"]:
                continue
            
            action = actions.get(name,0)
            #check which action and perform it
            if action==1:
                ship.vr -= self.turn_accel
            elif action ==2:
                ship.vr += self.turn_accel
            elif action ==3:
                #accelerates in the direction currently facing
                ship.vx += self.ship_accel *math.cos(ship.angle)
                ship.vy += self.ship_accel *math.sin(ship.angle)
            elif action==4:
                #fire a bullet if it has cooled down and still has bullets
                if ship.cooldown ==0 and len([b for b in self.bullets if b.owner == name])<self.max_bullets:
                    bx=ship.x+20*math.cos(ship.angle)
                    by=ship.y+20*math.sin(ship.angle)
                    bvx=self.bullet_speed*math.cos(ship.angle)+ship.vx
                    bvy=self.bullet_speed*math.sin(ship.angle)+ship.vy
                    #create nuew bullet object with agent's name
                    self.bullets.append(Bullet(bx,by,bvx,bvy,name))
                    ship.cooldown=20#wait 1 second
        #perform world physics
        for name, agent in self.agents.items():
            ship = agent["ship"]
            #move according to velocity
            ship.x+=ship.vx
            ship.y+=ship.vy
            #loop around edges
            ship.x%=self.world_size
            ship.y%=self.world_size
            ship.angle+=ship.vr
            #dampen rotation (to avoid spinning uncontrollably)
            ship.vr*=0.9
            ship.angle%=2*math.pi#loop angle
            #decrement cooldown
            if ship.cooldown>0:
                ship.cooldown-=1
        hits=[]
        alive_names = set(self.agents.keys())
        #check each bullet for any collisions after performing physics
        for b in self.bullets:
            #move
            b.x+=b.vx
            b.y+=b.vy
            #loop around edges
            b.x%= self.world_size
            b.y%= self.world_size
            #check for overlap with each agent
            for name in alive_names:
                ship = self.agents[name]["ship"]
                if b.owner == name:
                    continue
                #if within 15, ship dies
                if (b.x-ship.x)**2+(b.y-ship.y)**2<625:
                    self.agents[name]["alive"] = False
                    hits.append((b.owner, name))
        #only keep bullets of living agent
        self.bullets=[b for b in self.bullets if self.agents[b.owner]["alive"]]
        #reward each agent for current gamestate
        rewards = {name: 0.0 for name in self.agents}
        #major reward for hits
        for shooter, victim in hits:
            rewards[shooter]+=1.0
            rewards[victim]-=1.0

        obs={name: self.build_obs(name) for name in self.agents}
        if madrl:
            #additional rewards are given to solve sparse rewards
            for name in self.agents:
                #slight reward for facing enemy
                rewards[name]+=0.2*obs[name][9]
                #penalty for enemy facing you
                rewards[name]-=0.22*obs[name][11]
                #higher award for being close and facing enemy
                if obs[name][10] < 150 and obs[name][9] > 0.8:
                    rewards[name] += 0.3
                #strong penalty for enemy being close and facing you
                if obs[name][10] < 150 and obs[name][11] > 0.8:
                    rewards[name] -= 0.35
                #incentivise ending the match
                rewards[name]-= 0.01
                #slightly penalize rapid spinning
                rewards[name] -= 0.002*abs(obs[name][4])
            #penalize being too close to bullets
            for i in range(self.max_bullets):
                dx=obs[name][13+4*i]
                dy=obs[name][14+4*i]
                dist = (dx*dx+dy*dy)/0.007
                rewards[name]-=1/dist
        dones = {name: not self.agents[name]["alive"] for name in self.agents}

        if self.render_enabled:
            self._render()
        self.step_count+=1
        return obs, rewards, dones

    #will periodically render game
    def _render(self):
        for event in pygame.event.get():
            if event.type ==pygame.QUIT:
                exit()
        self.screen.fill((0,0,10))
        #draw each agent
        for name, agent in self.agents.items():
            ship = agent["ship"]
            color = (0,255,0) if name == "green" else (0,180, 255)
            pygame.draw.circle(self.screen, color, (int(ship.x), int(ship.y)),15)
            pygame.draw.circle(self.screen, color, (int(ship.x+15*math.cos(ship.angle)),int(ship.y+15*math.sin(ship.angle))),6)
        #draw each bullet
        for b in self.bullets:
            pygame.draw.circle(self.screen, (255,255,0), (int(b.x), int(b.y)), 4)
        pygame.display.flip()
        #wait for next frame
        time.sleep(1/self.fps)
