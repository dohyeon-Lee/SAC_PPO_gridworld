import gym
import numpy as np
from collections import defaultdict
import sys, os
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("env"))))
from env import gridworld

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        #####
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        #####
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo control function that uses epsilon-greedy policy
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Save the sum and the counts of the state.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # Nested dictionary that maps state -> (action -> action-value)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Generate an epsilon greedy policy
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #####
        state = env.reset()
        Timestep = []
        
        #while True:
        for t in range(100):
            prob = policy(state)
            action = np.random.choice(np.arange(len(prob)), p=prob)
            next_state, R, done, _ = env.step(action)
            G_member = [state, action, R]
            Timestep.append(G_member)
            if done == True:
                break
            # if i_episode > 997:
            #     env._render()
            #     time.sleep(1)

            state = next_state
        #first visit MC
        firstvisit_state = []
        # for calculate G
        for i in range(len(Timestep)):
            G = 0
            now_state_action = Timestep[i][:2]
            if now_state_action not in firstvisit_state:
                firstvisit_state.append(now_state_action)
                for j in range(len(Timestep)-i-1):
                    G = discount_factor*G + Timestep[-(j+1)][2]
                returns_sum[now_state_action[0],now_state_action[1]] += G
                returns_count[now_state_action[0],now_state_action[1]] += 1
        
        # for calculate Q
        state_check = []
        for i in returns_sum:
            Q[i[0]][i[1]] = returns_sum[i]/returns_count[i]

        #####    
        
    return Q, policy

def main():
    env = gridworld.GridworldEnv(10,10)
    
    Q, policy = mc_control_epsilon_greedy(env, num_episodes=1000)
    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    V = np.array([ V[s] for s in range(env.nS)])
    
    policy_table = np.array([policy(s) for s in range(env.nS)])
    # Policy Iteration
    print("--------Monte Carlo Control---------")
    print(policy_table)

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy_table, axis=1), env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(V.reshape(env.shape))

    print("")
    print("---------------------------------")

    
if __name__=="__main__":
    main()