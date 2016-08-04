function [W, mu] = fishertrain(trainset, trainlabels, c)
  [N, ~] = size(trainset);
  
  [~, W, mu] = eigentrain(trainset', N - c);
  rtrainset = trainset';
  for i = 1:N
      rtrainset(:,i) = rtrainset(:,i)-mu';
  end
  rtrainset = (W*rtrainset')';
  rmu = mean(rtrainset,2);
  
  classes = unique(trainlabels);
  Sw = 0;
  Sb = 0;
  for i = 1:c
    curr_class = rtrainset(:, trainlabels == classes(i));
    csize = size(curr_class, 2);
    cmu = mean(curr_class,2);
    B = (cmu - rmu); % between class
    Sb = Sb + csize*B*B';
    S = 0; % within class
    for j = 1:csize
      A = (curr_class(:,j) - cmu);
      S = S + A*A';
    end
    Sw = Sw + S;
  end
  [U, V] = eig(Sb, Sw);
  eigvals = diag(V);
  [sv, si] = sort(eigvals, 'descend');
  Wfld = U(:,si);
  Wfld = Wfld(1:c-1,:);
  %W = Wfld*W;
end
