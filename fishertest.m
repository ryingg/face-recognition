function testLabels = fishertest(trainset, trainlabels, testset, W, mu, k)

% multiply the W to train and testset
rtrainset = (W*trainset')';
rtestset = (W*testset')';

% perform 1NN L2 on reduced dimension set
[n, ~] = size(testset);
neighbors = zeros(n, 1);
for i = 1:n
    dist = sqrt(sum((rtrainset-repmat(rtestset(i,:),size(rtrainset,1),1)).^2,2));
    [~, index] = sort(dist);
    neighbors(i, :) = index(1:1);
end

testLabels = trainlabels(neighbors);
end

