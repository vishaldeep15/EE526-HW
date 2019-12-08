clear
clc
n=26; a=zeros(n);
a(1,2)=5;a(1,3)=6;a(1,4)=4;a(1,5)=7;
a(2,6)=2;a(2,7)=3;
a(3,6)=6;a(3,7)=4;a(3,8)=1;
a(4,7)=7;a(4,8)=3;a(4,9)=6;
a(5,8)=9;a(5,9)=1;
a(6,10)=2;a(6,11)=3;
a(7,10)=6;a(7,11)=4;a(7,12)=1;
a(8,11)=7;a(8,12)=3;a(8,13)=6;
a(9,12)=9;a(9,13)=7;
a(10,14)=2;a(10,15)=3;
a(11,14)=6;a(11,15)=4;a(11,16)=1;
a(12,15)=7;a(12,16)=10;a(12,17)=6;
a(13,16)=9;a(13,17)=8;
a(14,18)=2;a(14,19)=3;
a(15,18)=6;a(15,19)=4;a(15,20)=1;
a(16,19)=7;a(16,20)=3;a(16,21)=6;
a(17,20)=9;a(17,21)=1;
a(18,22)=2;a(18,23)=3;
a(19,22)=6;a(19,23)=4;a(19,24)=1;
a(20,23)=7;a(20,24)=3;a(20,25)=6;
a(21,24)=9;a(21,25)=1;
a(22,26)=5;
a(23,26)=6;
a(24,26)=4;
a(25,26)=7;


a=a+a';
for i=22:25
    d(i)=a(i,26);
end


for i=1:4
%     r=zeros(4,4);
    aa=[a(i+17,22)+d(22),a(i+17,23)+d(23),a(i+17,24)+d(24),a(i+17,25)+d(25)];
    [d(i+17)]=max(aa);
    
    p=find(aa==max(max(aa)));
    m=size(p,2);
    r(i+17,1:m)=p;
    [m,n]=size(r);
    for i=m
        for j=1:n
            if r(i,j)~= 0
                r(i,j)=r(i,j)+21;
            end
        end
    end
end





for i=1:4
    aa=[a(i+13,18)+d(18),a(i+13,19)+d(19),a(i+13,20)+d(20),a(i+13,21)+d(21)];
    [d(i+13)]=max(aa);
%     pp(i+13)=p(i)+17;
    p=find(aa==max(max(aa)));
    m=size(p,2);
    r(i+13,1:m)=p;
    [m,n]=size(r);
end
    for k=14:17
        for j=1:n
            if r(k,j)~= 0
                r(k,j)=r(k,j)+17;
            
            end
        end
    end


for i=1:4
    aa=[a(i+9,14)+d(14),a(i+9,15)+d(15),a(i+9,16)+d(16),a(i+9,17)+d(17)];
    [d(i+9)]=max(aa);

    p=find(aa==max(max(aa)));
    m=size(p,2);
    r(i+9,1:m)=p;
    [m,n]=size(r);
end
    for k=10:13
        for j=1:n
            if r(k,j)~= 0
                r(k,j)=r(k,j)+13;
            
            end
        end
    end

for i=1:4
    aa=[a(i+5,10)+d(10),a(i+5,11)+d(11),a(i+5,12)+d(12),a(i+5,13)+d(13)];
    [d(i+5)]=max(aa);

    p=find(aa==max(max(aa)));
    m=size(p,2);
    r(i+5,1:m)=p;
    [m,n]=size(r);
end
    for k=6:9
        for j=1:n
            if r(k,j)~= 0
                r(k,j)=r(k,j)+9;
            
            end
        end
    end

 for i=1:4
    aa=[a(i+1,6)+d(6),a(i+1,7)+d(7),a(i+1,8)+d(8),a(i+1,9)+d(9)];
    [d(i+1)]=max(aa);

    p=find(aa==max(max(aa)));
    m=size(p,2);
    r(i+1,1:m)=p;
    [m,n]=size(r);
end
    for k=2:5
        for j=1:n
            if r(k,j)~= 0
                r(k,j)=r(k,j)+5;
            
            end
        end
    end  
    
for i=1
    aa=[a(i,2)+d(2),a(i,3)+d(3),a(i,4)+d(4),a(i,5)+d(5)];
    d(i)=max(aa);
    p=find(aa==max(max(aa)));
    m=size(p,2);
    r(i,1:m)=p;
    [m,n]=size(r);
end
for i=1:n
    if r(1,i)~=0
        r(1,i)=r(1,i)+1;
    end
end
d(1)