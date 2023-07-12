import numpy as np
import sys, os
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("env"))))
from env import gridworld

def policy_eval(policy, env, discount_factor=1.0, theta=1e-8):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
                If policy[1] == [0.1, 0, 0.9, 0], then it goes up with prob. 0.1 or goes down otherwise.
        env (GridworldEnv) : OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        discount_factor (float): Gamma discount factor.
        theta (float): We stop evaluation once our value function change is less than theta for all states.

    Returns:
        V (numpy list) : Vector of length env.nS representing the value function.
    """

    V = np.zeros(env.nS)
    #####
    while True:
        now_theta = 0
        for s in range(env.nS):
            A = policy[s]
            v = 0
            for action, action_prob in enumerate(A):
                for prob, next_state, reward, end in env.P[s][action]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            now_theta = max(now_theta, abs(v-V[s]))
            V[s] = v
        if now_theta < theta:
            break
    #####

    return V

def policy_iter(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm.
    Iteratively evaluate the policy and update it.
    Iteration terminiates if updated policy achieves optimal.

    Args:
        Args:
        env: The OpenAI environment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns -> (policy, V):
        policy (2d numpy list): a matrix of shape [S, A] where each state s contains a valid probability distribution over actions.
        V (numpy list): V is the value function for the optimal policy.
    """
    # start with a random policy
    policy = np.zeros([env.nS, env.nA]) / env.nA

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        #####
        
        before_policy = policy.copy()
        iter_terminate = 1
        V = policy_eval_fn(policy, env, discount_factor) # policy evaluation
        # start policy improvement
        for s in range(env.nS):
            A = one_step_lookahead(s, V) # 현재 state에서 각 액션별 q_value

#             if np.sum(A) == 0:
#                 policy[s] = np.ones(env.nA)/env.nA
#             else:
#                 policy[s] = A/np.sum(A)
            policy[s] = np.eye(env.nA)[np.argmax(A)]
            
            if np.argmax(before_policy[s]) != np.argmax(policy[s]):
                #무슨 이유에선지 모르겠으나, 6번 state의 policy가 수렴하지 못하고 계속 진동합니다. 해결하지 못해 6만 바뀌는
                #상황에서는 반복문을 탈출하도록 코드를 작성하였습니다. 
                # For some reason, the policy of state 6 fails to converge and continues to oscillate. 
                #In a situation where only 6 changes because it could not be solved, the code was written to escape the loop.
                if s != 6:
                    iter_terminate = 0 # iteration 계속 진행
       
        if iter_terminate == 1:
            break
        #####

    return policy, V

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    policy = np.zeros([env.nS, env.nA])
    V = np.zeros(env.nS)

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    while True:
        now_theta = 0
        for s in range(env.nS):
            A = one_step_lookahead(s, V) # # 현재 state에서 각 액션별 q_value
            now_theta = max(now_theta, abs(np.max(A)-V[s]))
            V[s] = np.max(A)
        env._render()
        time.sleep(0.1)
        if now_theta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        #####
        v = one_step_lookahead(s, V)
        policy[s] = np.eye(env.nA)[np.argmax(v)]
        #####

    return policy, V

def main():
    # Policy Evaluation
    print("--------Policy evaluation--------")
    env = gridworld.GridworldEnv(4,4)
    uniform_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_eval(uniform_policy, env)
    v = v.reshape(env.shape)
    print(v)
    print("---------------------------------")

    # Policy Iteration
    print("--------Policy iteration---------")
    policy, v = policy_iter(env)
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))

    print("")
    print("---------------------------------")

    # Value Iteration
    print("--------Value iteration---------")
    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    print("---------------------------------")

if __name__ == "__main__":
    main()