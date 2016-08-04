function [ successes, found ] = eigenaccuracy(trainset, testset, trainlabels, testlabels, e)

[W, allW, mu] = eigentrain(trainset,20);

allW = allW(:,e:end); % cut off first e

[n,~] = size(testset);
toplot = zeros(20,1);
for k = 1:20
    W = allW(:,1:k)'; % get w for first k
    accuracy = 0;
    found = eigentest(trainset, trainlabels, testset, W, mu, k);
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

