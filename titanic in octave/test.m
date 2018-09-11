data=load('train_modified.csv');
X=data(:,1:6);
y=data(:,7);

xfare=X(:,6);
bincnt=zeros(1,50);

for i=1:length(X)
  if (xfare(i)<=50 && y(i)==1)
    bincnt(1)=bincnt(1)+1;
  elseif (xfare(i)>50 && xfare(i)<=100 && y(i)==1)
    bincnt(2)=bincnt(2)+1; 
  elseif (xfare(i)>100 && xfare(i)<=150 && y(i)==1)
    bincnt(3)=bincnt(3)+1; 
  elseif (xfare(i)>150 && xfare(i)<=200 && y(i)==1)
    bincnt(4)=bincnt(4)+1;  
  elseif (xfare(i)>200 && xfare(i)<=250 && y(i)==1)
    bincnt(5)=bincnt(5)+1; 
  elseif (xfare(i)>250 && xfare(i)<=300 && y(i)==1)
    bincnt(6)=bincnt(6)+1; 
  elseif (xfare(i)>300 && xfare(i)<=350 && y(i)==1)
    bincnt(7)=bincnt(7)+1; 
  elseif (xfare(i)>350 && xfare(i)<=400 && y(i)==1)
    bincnt(8)=bincnt(8)+1;
  elseif (xfare(i)>400 && xfare(i)<=450 && y(i)==1)
    bincnt(9)=bincnt(9)+1; 
  elseif (xfare(i)>450 && xfare(i)<=500 && y(i)==1)
    bincnt(10)=bincnt(10)+1; 
  elseif (xfare(i)>500 && xfare(i)<=550 && y(i)==1)
    bincnt(11)=bincnt(11)+1;
  endif;
end

hist(bincnt,0:50:550,"facecolor", "c", "edgecolor", "b");


