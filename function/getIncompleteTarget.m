
function [target, totoalNum, totoaldeleteNum, realpercent,obsTarget_index]= getIncompleteTarget(target, percent, bQuiet)
% delete the elements in the target matrix 'oldtarget' by the given percent
% oldtarget : N by L data matrix
% percent   : 10%, 20%, 30%, 40%, 50%
obsTarget_index = ones(size(target));

totoalNum = sum(sum(target ==1));
totoalNum = totoalNum + sum(sum(target ==-1)); %counting -1s, Sanjay
totoaldeleteNum = 0;
[N,~] = size(target);
realpercent = 0;
maxIteration = 50;
factor = 2;
count=0;
while realpercent < percent
    if maxIteration == 0
        factor = 1;
        maxIteration = 10;
        if count==1
            break;
        end
        count = count+1;
    else
        maxIteration = maxIteration - 1;
    end
    for i=1:N
        %index = find(target(i,:)==1);
        index = find(target(i,:)==1 | target(i,:)==-1); %observed labels
        %mean 1 as well as -1, sanjay
        if length(index) >= factor % can be set to be 1 if the real missing rate can not reach the pre-set value
            deleteNum = round(1 + rand*(length(index)-1));%至少保证该样本有个类别标签
            totoaldeleteNum = totoaldeleteNum + deleteNum;
            realpercent = totoaldeleteNum/totoalNum;
            if realpercent >= percent
                break;
            end
            
            index = index(randperm(length(index)));
                target(i,index(1:deleteNum)) = 0;
                obsTarget_index(i,index(1:deleteNum))=0;
            end
        end
    end
end


