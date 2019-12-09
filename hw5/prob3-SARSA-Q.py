import numpy as np 
    
def epsilon_greedy(q, epsilon, state):
    if np.random.rand() < epsilon:
        action = np.random.randint(2)
    else:
        action = np.argmax(q[state, :])
    return action

def logQ(iteration):
    print(f"Iteration: {iteration}")
    print(f"q(0,0): {q[0,0]:.4f}, q(0,1): {q[0,1]:.4f}, q(1,0): {q[1,0]:.4f}, q(1,1): {q[1,1]:.4f}")
    
def init_params():
    q = np.random.rand(2, 2)
    state = np.random.randint(2)
    action = epsilon_greedy(q, epsilon, state)
    rewards = np.array([[1., 4.],
                        [3., 2.]])
    return q, state, action, rewards

epsilon = 0.1
alpha = 0.1
gamma = 3/4

# intialize Q with random numbers
(q, state, action, rewards) = init_params()

print("\n=========================SARSA Policy=========================")
for i in range(100000):
    # get next state
    state_next = np.random.randint(2)
    # get next action
    action_next = epsilon_greedy(q, epsilon, state_next)
    # get reward
    R = rewards[state, action]

    # update q
    q[state, action] = q[state, action] + alpha * (R +  gamma*q[state_next, action_next] - q[state, action])

    state = state_next
    action = action_next
    
    if i%10000 == 0:
        logQ(i)


        
(q, state, action, rewards) = init_params()
print("\n=========================Q-Learning Policy=========================")  
for i in range(100000):
    # get next state
    state_next = np.random.randint(2)
    # get next action
    action_next = epsilon_greedy(q, epsilon, state_next)
    # get reward
    R = rewards[state, action]

    # update q
    q[state, action] = q[state, action] + alpha * (R +  gamma*np.max(q[state_next, :]) - q[state, action])

    state = state_next
    action = action_next
    
    if i%10000 == 0:
        logQ(i)

