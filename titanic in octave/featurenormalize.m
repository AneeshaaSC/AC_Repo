function [X] = featurenormalize(X)
len=length(X)

class1=find(X(:,7)==1);
class2=find(X(:,7)==2);
class3=find(X(:,7)==3);

class1_avg=mean(X(class1,7));
class2_avg=mean(X(class2,7));
class3_avg=mean(X(class3,7));

ze=find(X(:,7)==0); 
zl=length(ze);


for i=1:zl
  if X(7,ze)==0 && X(1,ze)==1
    X(7,ze)=class1_avg;
  elseif X(7,ze)==0 && X(1,ze)==2
    X(7,ze)=class1_avg;
  elseif X(7,ze)==0 && X(1,ze)==3
    X(7,ze)=class1_avg;    
  endif;

end;
