function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

dis_X_all = zeros(size(X,1),K);

for i = 1:K
    dev_X = X .- centroids(i,:);
    dis_X = zeros(size(X,1),1);
    for dim = 1:size(X,2)
        dis_X = dis_X .+ dev_X(:,dim).^2;
    end
    dis_X_all(:,i) = dis_X;
end
[_, idx_t] = min(dis_X_all');
idx = idx_t';

% =============================================================

end

