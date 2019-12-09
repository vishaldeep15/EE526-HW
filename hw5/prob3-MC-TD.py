import numpy as np

def setStates(nStates):
    # Generates 10000 random states between 0 to 1
    states = np.random.randint(0, 2, size=nStates)
    # Make sure 1st state is 0
    states[0] = 0
    return states

def setRewards(states):
    # Initialize reward array
    rewards = np.zeros(states.shape)
    # policy π that chooses action 1 in state 0 => Reward = 1, 
    # and action 2 in state 1 => Reward = 2
    rewards = updateRewards(states)
    return rewards

def Gt(states, rewards, gamma):
    G_t = 0
    # Calculate discounted rewards
    for i in range(1, len(states)):
        G_t += rewards[i] * np.power(gamma, i-1) 
    return G_t

def updateRewards(states):
    for idx, state in enumerate(states):
        if state == 0:
            rewards[idx] = 1
        elif state == 1:
            rewards[idx] = 2
    return rewards

def getNextValue(iteration, n, states):
    if states[iteration+n] == 0:
        V_next = V0
    elif states[iteration+n] == 1:
        V_next = V1
    return V_next

def nStepTD(states, rewards, gamma, n, V0, V1, alpha=0.1):
    for i in range(len(states)-n): 
        if states[i] == 0:
            # Get discounted rewards
            Gt_n = Gt(states[i:i+n+1], rewards[i:i+n+1], gamma)
            # Get n-step future value
            V_next = getNextValue(i, n, states)
            # Estimate value function
            V0 = V0 + alpha * (Gt_n + np.power(gamma, n)*V_next - V0)
        else:
            # # Get discounted rewards
            Gt_n = Gt(states[i:i+n+1], rewards[i:i+n+1], gamma)
            # Get n-step future value
            V_next = getNextValue(i, n, states)
            # Estimate value function
            V1 = V1 + alpha * (Gt_n + np.power(gamma, n)*V_next - V1)
    return V0, V1

nStates = 10000
gamma = 3/4
alpha = 0.1
V0, V1, N0, N1 = 0, 0, 0, 0

states = setStates(nStates)
rewards = setRewards(states)

print("==============Monte Carlo Policy===============")
# Monte carlo policy
# Iterate through each episode
for i in range(len(states)):
    if states[i] == 0:
        N0 += 1
        # Get discounted rewards
        G_t = Gt(states[i:], rewards[i:], gamma)
        # Calculate value function with incremental mean
        V0 = V0 + 1/N0 * ( G_t - V0 )
    elif states[i] == 1:
        N1 += 1
        # Get discounted rewards
        G_t = Gt(states[i:], rewards[i:], gamma)
        # Calculate value function with incremental mean
        V1 = V1 + 1/N1 * ( G_t - V1 )
        
    if i%999 == 0:
        print(f"Iteration: {i}, V0: {V0:.4f}, V1: {V1:.4f}")

# Initialize again for TD Policy
V0, V1, N0, N1 = 0, 0, 0, 0

print("=================TD Policy====================")
# n-step TD Policy where n is 1 to 10
for n in range(1,11):
     (V0, V1) = nStepTD(states, rewards, gamma, n, V0, V1)
     print(f"n-step: {n}, V0: {V0:.4f}, V1: {V1:.4f}")
