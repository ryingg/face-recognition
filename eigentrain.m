function [W, allW, mu] = eigentrain (trainset, k)
trainset = trainset';  
mu = mean(trainset, 1);

trainset = trainset - repmat(mu, size(trainset,1),1);
sigma = trainset*trainset'/size(trainset,2);
[~, ~, V] = svd(sigma); % eig sigma;
W = V(:, 1:k);
W = W';
allW = V;

end
