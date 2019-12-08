import numpy as np

def q_pi(s, a, V0, V1):
    gamma = 3/4
    
    if s==0 and a==1:
        R = 1
        P0 = 1/3
        P1 = 1 - P0
    elif s==0 and a==2:
        R = 4
        P0 = 1/2
        P1 = 1- P0
    elif s==1 and a==1:
        R = 3
        P0 = 1/4
        P1 = 1 - P0
    elif s==1 and a==2:
        R = 2
        P0 = 2/3
        P1 = 1- P0
    
    v_star = R + gamma * (P0*V0 + P1*V1)
    return v_star

# Intialize to zero
V0 = 0
V1 = 0
    
for i in range(0, 10):
    V0_temp = np.maximum(q_pi(0, 1, V0, V1), q_pi(0, 2, V0, V1))
    V1 = np.maximum(q_pi(1, 1, V0, V1), q_pi(1, 2, V0, V1))
    V0 = V0_temp
    print(f"Iteration: {i}, V0: {V0:.4f}, V1: {V1:.4f}")

q01 = q_pi(0, 1, V0, V1)
q02 = q_pi(0, 2, V0, V1)
q11 = q_pi(1, 1, V0, V1)
q12 = q_pi(1, 2, V0, V1)

print(f"q(0,1) = {q01:.4f}")
print(f"q(0,2) = {q02:.4f}")
print(f"q(1,1) = {q11:.4f}")
print(f"q(1,2) = {q12:.4f}")