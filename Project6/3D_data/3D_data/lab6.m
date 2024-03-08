fig1=figure('Name','3d_sphere.mat','NumberTitle','off');
load('3d_sphere.mat');
[three_d_sphere_cov_matrix,three_d_sphere_eigenvalues,three_d_sphere_eigenvectors] = PCA(X);

plot3d_pca(X);

fig2=figure('Name','teapot.mat','NumberTitle','off');
load('teapot.mat');
[teapot_cov_matrix,teapot_eigenvalues,teapot_eigenvectors] = PCA(X);

plot3d_pca(X);

fig3=figure('Name','bun_zipper.mat','NumberTitle','off');
load('bun_zipper.mat');
[bun_zipper_cov_matrix,bun_zipper_eigenvalues,bun_zipper_eigenvectors] = PCA(X);

plot3d_pca(X);
function [X_centered, centroid] = center(data)
    % mean in d-dimension
    centroid = mean(data, 1);
    
    % center the points by using data - mean_data
    X_centered = data - centroid;
end
%calcualate the co-variance
function [co_var, values, vectors] = PCA(X)
    %co-variance is 1/n*(X-mean)(X-mean)^T
    [X_centered,mean] = center(X);
    [N,d] = size(X);
    co_var = (1/N)*(X_centered)*(X_centered');
    %Get the eigenvalue,eigenvectors, vectors contain eignevectors as col.
    %And values contains the eigenvalues on the diagonal elements
    [vectors,values] = eig(co_var);
end


function plot3d_pca(X)
    %plot the X first
    plot(X(:,1),X(:,2),X(:,3));
    hold on;
    %get the scale by using its co_variance
    [co_var, values, vectors] = PCA(X);
    for  i =1:3
        scale = co_var(i);

    end




end