function [ successes, found ] = fisheraccuracy(trainset, testset, trainlabels, testlabels, W)

[n,~] = size(testset);
toplot = zeros(20,1);
for k = 1:20
    accuracy = 0;
    found = fishertest(trainset, trainlabels, testset, W, mu, k);
    for i = 1:n
        if found(i) == testlabels(i) %check if accurate
            accuracy = accuracy + 1;
        end
    end
    accuracy = accuracy/n;
    toplot(k) = 1 - accuracy;
end

successes = toplot;
end

