function [neighbors, accuracy] = knn(trainset, testset, trainlabels, testlabels, k, l)

[n, ~] = size(testset);
neighbors = zeros(n, k);
accuracy = 0;

for i = 1:n
    if l == 1 %l1 norm manhattan
      dist = sum(abs(trainset - repmat(testset(i,:),size(trainset,1),1)),2);
    else %l2 norm euclid
      dist = sqrt(sum((trainset-repmat(testset(i,:),size(trainset,1),1)).^2,2));
    end
    [~, index] = sort(dist); % k neighbors
    neighbors(i, :) = mode(trainlabels(index(1:k))); %calc neighbors with mode
    if (neighbors(i,:) == testlabels(i)) %check if accurate
      accuracy = accuracy + 1; %increment num of accurate
    end
end

accuracy = accuracy/n;

end