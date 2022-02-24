import gym
import time
import pickle
from tqdm import tqdm
import numpy as np



alfa = 0.1
gamma = 0.6
epsilon = 0.01

env = gym.make("Taxi-v3")
env.reset()
epochs = 15_000

Q = np.zeros([env.observation_space.n , env.action_space.n])


for i in tqdm(range(epochs)):
    done = False
    new_state = env.reset()
    while(not done):
        action = env.action_space.sample()
        #action = np.argmax(Q[new_state , :])
        valor_viejo = Q[new_state , action]
        estado_viejo = new_state # esto hay que guardarlo
        new_state , reward,  done , info = env.step(action)
        valor_maximo = np.max(Q[new_state])
        new_Q = (1 - alfa) * valor_viejo + alfa*(reward + (gamma  * valor_maximo))
        Q[estado_viejo , action ] = new_Q
        estado_viejo = new_state


with open(f"Q_matrix_trained_{epochs}_on_{time.time()}.pkl","wb") as file:
    pickle.dump(Q,file)


wins = 0
print("testing...")
for i in tqdm(range(1000)):
    done = False
    new_state = env.reset()
    while(not done):
        action = np.argmax(Q[new_state])
        new_state, reward , done , info = env.step(action)
        env.render()
        if(reward > -1):
            wins += 1

print(wins)
