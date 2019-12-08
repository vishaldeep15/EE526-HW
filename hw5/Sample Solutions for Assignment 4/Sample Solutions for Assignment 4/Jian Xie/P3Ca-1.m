

clear;
clc
close all
Q = [1,3;1,2];%initial Q

R = [1, 4; 3, 2];
state = 0;
epsilon = 0.12;
alpha = 0.1;
gamma = 3/4;


iter_times = 10000;
rand('seed',0);
P = rand(1,iter_times)';

P_temp = rand(1,1);
if P_temp<epsilon      % randomly take actions
    a = randi([1 2],1,1);  %%action
else
%     a=max(Q(state+1,:));     % take action according to Q chart
    [~, a] = max(Q(state+1, :));
end

for j = 1:iter_times
%     state=s-1;
    if state == 0   %%% state 0
        if a == 1    %%action 1
           P_temp = rand(1,1);
           if P_temp>=1/3   
               state_prime = 1;
           else
               state_prime = 0;
           end
        else 
           P_temp = rand(1,1);
           if P_temp>=1/2
               state_prime = 1;
           else
               state_prime = 0;
           end
        end
    else
        P_temp = rand(1,1);
        if a == 1
            
           if P_temp>=1/4
               state_prime = 1;
           else
               state_prime = 0;
           end
        else
            P_temp = rand(1,1);
            if P_temp>=2/3
               state_prime = 1;
            else
                state_prime = 0;
            end
        end
    end

    
    reward = R(state+1, a);
    
    P_temp = P(j);  %% current state is state_prime
    if P_temp<epsilon      % randomly take actions
        a_prime = randi([1 2],1,1);  %%action
    else
%         a_prime=max(Q(state_prime+1,:));     % take action according to Q chart
        [~, a_prime] = max(Q(state_prime+1, :));
    end
    
    Q(state+1,a) = Q(state+1,a) + alpha* (reward + gamma*Q(state_prime+1, a_prime) - Q(state+1,a));
    state = state_prime; 
    a = a_prime;
    
    Q_s0_a1(j) = Q(1,1);
    Q_s0_a2(j) = Q(1,2);
    Q_s1_a1(j) = Q(2,1);
    Q_s1_a2(j) = Q(2,2);
 
end
Q_prime=[Q_s0_a1; Q_s0_a2; Q_s1_a1; Q_s1_a2];
figure;
plot(Q_prime(1,:));
hold on;
plot(Q_prime(2,:));
hold off;
xlabel('iteration times');
ylabel('Q(s0, action)');
legend('Q(s0,a1)', 'Q(s0,a2)');


figure;
plot(Q_prime(3,:));
hold on;
plot(Q_prime(4,:));
hold off;
xlabel('iteration times');
ylabel('Q(s1, action)');
legend('Q(s1,a1)', 'Q(s1,a2)');






