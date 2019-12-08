clear;
clc
close all
gamma = 3/4;
iter=10000;
RSA(1, :) = [0, 0];
%% (a)
rand('seed',1);

P_r= rand(1,10000)';
for j = 1:iter
    P = P_r(j);
    [m,n] = size(RSA);
    if RSA(m,2) == 1
       if P > 1/3   % from state 1 transfer to state 0
          RSA(m,3) = 2;  % action 2
          RSA(m,1) = 2;
          RSA(m+1,2) = 0;  % sate 0
       elseif P<=1/3   % from state 1 transfer to state 1
          RSA(m,3) = 2;
          RSA(m,1) = 2;
          RSA(m+1,2) = 1;  % sate 1  
       end
    else
       if P > 1/3   % from state 0 transfer to state 1
          RSA(m,3) = 1;  % action 1
          RSA(m,1) = 1;
          RSA(m+1,2) = 1;  % sate 1
       elseif P<=1/3   % from state 0 transfer to state 0
          RSA(m,3) = 1;
          RSA(m,1) = 1;
          RSA(m+1,2) = 0;  % sate 0           
       end         
    end
end

%% (b)



G=zeros(iter,1);
for j=1:iter
    temp_R=RSA(j:iter,1);
    mm=length(temp_R);
    temp=[];
    for i=1:mm
        temp(i)=gamma^(i-1)*temp_R(i);
    end
    G(j)=G(j)+sum(temp);
end
indx_s0 = find(RSA(:,2)==0);
indx_s0(end, :) = [];
G_s0 = G(indx_s0);

V_s0=zeros(length(G_s0),1);
for j = 1:length(G_s0)
    if j==1
        V_s0(j) = V_s0(j) + (G_s0(j,1)-V_s0(j))/j;
    else
        V_s0(j) = V_s0(j-1) + (G_s0(j,1)-V_s0(j-1))/j;
    end

end
indx_s1 = find(RSA(:,2)==1);
G_s1 = G(indx_s1);
V_s1=zeros(length(G_s1),1);
for j = 1:length(G_s1)
    if j==1
        V_s1(j) = V_s1(j) + (G_s1(j,1)-V_s1(j))/j;
    else
        V_s1(j) = V_s1(j-1) + (G_s1(j,1)-V_s1(j-1))/j;
    end

end
figure;
plot(V_s0);
hold on;
plot(V_s1);
title('Problem (b)');
xlabel('Iteration times');
ylabel('Value');
legend('V(0)','V(1)');
%% (c)
alpha = 0.02;
n_step_Value_function = {};
for n_step = 1:5
    V_s0 = 0;
    V_s1 = 0;
    for i = 1:(iter-n_step+1)

        R_S_A = RSA(i:i+n_step,:);
        m=size(R_S_A,1);
        for k = 1:m
            tg(k)=gamma^(k-1)*R_S_A(k,1);           
        end
        
        if R_S_A(end,2)==1
            tg(end) = gamma^(k-1)*V_s1;
        else
            tg(end) = gamma^(k-1)*V_s0;
        end
        
        G_temp = sum(tg);
        if RSA(i,2)==1
            V_s1 = V_s1 + alpha*(G_temp-V_s1);
        else
            V_s0 = V_s0 + alpha*(G_temp-V_s0);
        end 
        V_s0_c(i)=V_s0;
        V_s1_c(i)=V_s1;
    end
    n_step_Value_function{n_step,1} = V_s0_c;
    n_step_Value_function{n_step,2} = V_s1_c;

end


figure;
for i = 1:n_step
    plot(n_step_Value_function{i,1});
    hold on;
end
xlabel('iteration times');
ylabel('V(0)');
legend('n_step=1','n_step=2','n_step=3','n_step=4','n_step=5');



figure;
for i = 1:n_step
    plot(n_step_Value_function{i,2});
    hold on;
end
xlabel('iteration times');
ylabel('V(1)');
legend('n_step=1','n_step=2','n_step=3','n_step=4','n_step=5');