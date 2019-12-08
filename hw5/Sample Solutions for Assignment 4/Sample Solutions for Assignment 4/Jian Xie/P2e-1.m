clear
clc
V1=0;
V0=0;
V=[V0;V1];
r=3/4;
R0=[1;4];
R1=[3;2];
P0=[1/3 2/3;1/2 1/2];
P1=[1/4 3/4;2/3 1/3];
iter=50;
t=1:1:iter;
for i=1:iter
    q_state0(:,i)=R0+r*P0*V;
V(1)=max(q_state0(:,i));
q_state1(:,i)=R1+r*P1*V;
V(2)=max(q_state1(:,i));
VO(i,:)=V;
end
figure
plot(t,VO(:,1));
xlabel('iteration time')
ylabel('V0')
figure
plot(t,VO(:,2));
xlabel('iteration time')
ylabel('V1')