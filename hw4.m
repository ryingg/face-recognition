[trainset, trainlabels] = loadSubset(0);
[testset1, testlabels1] = loadSubset(1);
[testset2, testlabels2] = loadSubset(2);
[testset3, testlabels3] = loadSubset(3);
[testset4, testlabels4] = loadSubset(4);

%{
%1 set, labels, k, norm
[n1, a1] = knn(trainset, testset1, trainlabels, testlabels1, 1, 2);
[n2, a2] = knn(trainset, testset2, trainlabels, testlabels2, 1, 2);
[n3, a3] = knn(trainset, testset3, trainlabels, testlabels3, 1, 2);
[n4, a4] = knn(trainset, testset4, trainlabels, testlabels4, 1, 2);

%2 set, labels, k, norm
[n1, a1] = knn(trainset, testset1, trainlabels, testlabels1, 1, 1);
[n2, a2] = knn(trainset, testset2, trainlabels, testlabels2, 1, 1);
[n3, a3] = knn(trainset, testset3, trainlabels, testlabels3, 1, 1);
[n4, a4] = knn(trainset, testset4, trainlabels, testlabels4, 1, 1);

%3.1
[W, allW, mu] = eigentrain(trainset, 20);
%3.2
imshow(drawFaces(W,10),[]);

%3.4
mu = mean(trainset,1);
[a,~] = size(trainset);
temp = zeros(size(trainset));
reconstructed = zeros(size(trainset));
for k = 1:10
    for j=1:a
        temp(j,:) = trainset(j,:) - mu;
    end
    W = allW(:,1:k)';
    projected = (W*t')';
    projected = projected * W; 
    for j=1:a
        reconstructed(j,:) = projected(j,:) + mu;
    end
    subplot(10,1,k);
    imshow(drawFaces(final(1:7:end,:),10),[]);
    title(sprintf('k = %d',k));
end

%3.5
e = 1;
toplot = eigenaccuracy(trainset, testset1, trainlabels, testlabels1, e);
line([1:20],toplot,'Color',[0 0 0]);
toplot = eigenaccuracy(trainset, testset2, trainlabels, testlabels2, e);
line([1:20],toplot,'Color',[0 0 1]);
toplot = eigenaccuracy(trainset, testset3, trainlabels, testlabels3, e);
line([1:20],toplot,'Color',[0 1 0]);
toplot = eigenaccuracy(trainset, testset4, trainlabels, testlabels4, e);
line([1:20],toplot,'Color',[1 0 0]);
title('Error Rate All Eigenvectors');
legend('testset1','testset2','testset3','testset4');

%3.6
e = 5;
toplot = eigenaccuracy(trainset, testset1, trainlabels, testlabels1, e);
line([1:20],toplot,'Color',[0 0 0]);
toplot = eigenaccuracy(trainset, testset2, trainlabels, testlabels2, e);
line([1:20],toplot,'Color',[0 0 1]);
toplot = eigenaccuracy(trainset, testset3, trainlabels, testlabels3, e);
line([1:20],toplot,'Color',[0 1 0]);
toplot = eigenaccuracy(trainset, testset4, trainlabels, testlabels4, e);
line([1:20],toplot,'Color',[1 0 0]);
title('Error Rate Without First 4 Eigenvectors');
legend('testset1','testset2','testset3','testset4');

%}

%4.1
[W, mu] = fishertrain(trainset, trainlabels, 10);
%4.2
imshow(drawFaces(W,10),[]);

%4.3
toplot = fisheraccuracy(trainset, testset1, trainlabels, testlabels1, W);
line([1:20],toplot,'Color',[1 0 0]);
toplot = fisheraccuracy(trainset, testset2, trainlabels, testlabels2, W);
line([1:20],toplot,'Color',[0 0 1]);
toplot = fisheraccuracy(trainset, testset3, trainlabels, testlabels3, W);
line([1:20],toplot,'Color',[0 1 0]);
toplot = fisheraccuracy(trainset, testset4, trainlabels, testlabels4, W);
line([1:20],toplot,'Color',[1 0 1]);
title('Error Rate Fisher');
legend('testset1','testset2','testset3','testset4');
