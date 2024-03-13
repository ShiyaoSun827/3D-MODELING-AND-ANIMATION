fig1=figure();
data = load('3d_sphere.mat');
[co_var1, values1, vectors1] = PCA(data.X);
plot3d_pca(data.X);
fprintf('The Eigenvalue of  of 3d_sphere is\n');
disp(values1);
fprintf('The Eigenvector of  of 3d_sphere is\n');
disp(vectors1);


fig2=figure();
data2 = load('teapot.mat');
plot3d_pca(data2.X);
[co_var2, values2, vectors2] = PCA(data2.X);
fprintf('The Eigenvalue of  of teapot is\n');
disp(values2);
fprintf('The Eigenvector of  of teapot is\n');
disp(vectors2);

fig3=figure();
data3 = load('bun_zipper.mat');
plot3d_pca(data3.X);
[co_var3, values3, vectors3] = PCA(data3.X);
fprintf('The Eigenvalue of  of bun_zipper is\n');
disp(values3);
fprintf('The Eigenvector of  of bun_zipper is\n');
disp(vectors3);

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
    %size(X)
    s = scatter3(X(:,1),X(:,2),X(:,3));
    s.SizeData = 0.1;
    hold on;
    %get the scale by using its co_variance
    [co_var, values, vectors] = PCA(X);
    colors = ['b', 'r', 'g'];
    for  i = 1:3
        %size(vectors)
        scale = diag(values);
        scaled = sqrt(scale(i));
        %size(scaled)
        [X_centered, centroid] = center(X);
        
       
        quiver3(centroid(1), centroid(2), centroid(3), scaled*vectors(1,i), scaled*vectors(2,i), scaled*vectors(3,i), colors(i), 'LineWidth', 2);
    
        
    

    end
    hold off;
    xlabel('X_axis');
    ylabel('Y_axis');
    zlabel('Z_axis');
    title('3D');
    grid on;
    axis equal;




end