fig1=figure('Name','3d_sphere.mat','1','off');
data = load('3d_sphere.mat');

plot3d_pca(data.X);

fig2=figure('Name','teapot.mat','2','off');
data2 = load('teapot.mat');
plot3d_pca(data2.X);

fig3=figure('Name','bun_zipper.mat','3','off');
data3 = load('bun_zipper.mat');
plot3d_pca(data3.X);

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
    %[N,d] = size(X);
    %co_var = (1/N)*(X_centered)*(X_centered');
    co_var = cov(X_centered);
    %Get the eigenvalue,eigenvectors, vectors contain eignevectors as col.
    %And values contains the eigenvalues on the diagonal elements
    [vectors,values] = eig(co_var);
end


function plot3d_pca(X)
    %plot the X first
    size(X)
    scatter3(X(:,1),X(:,2),X(:,3));
    hold on;
    %get the scale by using its co_variance
    [co_var, values, vectors] = PCA(X);
    colors = ['b', 'r', 'g'];
    for  i =1:3
        
        scale = co_var(i);
        scaled = sqrt(scale)*vectors(:,i);
        [X_centered, centroid] = center(X);



  


        quiver3(centroid(1), centroid(2), centroid(3), scaled(1), scaled(2), scaled(3), colors(i), 'LineWidth', 2);
    
        
    

    end
    hold off;
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');
    title('3D PCA Visualization');
    grid on;
    axis equal;




end