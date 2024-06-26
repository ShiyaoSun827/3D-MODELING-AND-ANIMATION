%Question(A):
X = pcread('dragondata/dragon.ply');
figure(1)
hold on;
scatter3(X.Location(:,1), X.Location(:,2), X.Location(:,3),"b.");

%  ---------------- Write your code here  ---------------------------



hold off;

%Questioin(B)
% the number of points in the point cloud
N = size(X.Location, 1);


disp(['Number of points' num2str(N)]);

%Number of points: 437645

%  ------------------------------------------------------------------

%Question(C):
% Perform PCA on the point cloud data
coeff = pca(X.Location);

% The first eigenvector
firstEigenvector = coeff(:, 1);


disp('The first eigenvector:');
disp(firstEigenvector);

%The first eigenvector:
        % 0.8899
        %-0.4554
        % 0.0252

%  ------------------------------------------------------------------
%Question(D):
% The first eigenvector
firstEigenvector = coeff(:, 1);

% Normalized first eigenvector
v = firstEigenvector / norm(firstEigenvector);

% Since Transform the points of your model by a rotation, so that the First Eigen Vector becomes 
%the New X-axis.Then we use 1 0 0
x_axis = [1; 0; 0];

% Compute the axis of rotation (cross product of first eigenvector and x-axis)
axis_of_rotation = cross(v, x_axis);
%disp(axis_of_rotation);
% Compute the angle of rotation (using the dot product and inverse cosine)
angle_of_rotation = acos(dot(v, x_axis));




rotationMatrix = vrrotvec2mat([axis_of_rotation/ norm(axis_of_rotation); angle_of_rotation]);


%disp(rotationMatrix);

% Transform the point cloud
transformed_points = (rotationMatrix * X.Location')';

%  ------------------------------------------------------------------
%Question(E):
% Sort the transformed points following Part (d), based on the new X values.
[~, sortIdx] = sort(transformed_points(:, 1), 'ascend');
sortedTransformedPoints = transformed_points(sortIdx, :);

%  ------------------------------------------------------------------
%Question(F):
% Determine the Minimum, and Maximum, following Part (e).
Mini1 = min(sortedTransformedPoints(:, 1));
Max1 = max(sortedTransformedPoints(:, 1));


disp(['Minimum X value: ' num2str(Mini1)]);
disp(['Maximum X value: ' num2str(Max1)]);

%Minimum X value: -0.169
%Maximum X value: 0.039677
%  ------------------------------------------------------------------


% Question(G): Divide interval into 100 parts with approximately the same number of points
points = linspace(Mini1, Max1, 101);
cls_cen = cell(100, 1);  

%Reference:https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
for i = 1:100
    % Get the points in the current interval
    processed_data = transformed_points(:, 1) >= points(i) & transformed_points(:, 1) < points(i+1);
    data = transformed_points(processed_data, :);
    
    % QuestionH(a): Create N/1000 clusters
    Maximum = max(round(N/1000), 1);  % Ensure at least one cluster
    
    
    % QuestionH(b): Perform k-means clustering
    %Reference:https://www.mathworks.com/help/stats/kmeans.html
    
    [cluster_idx, cluster_centers] = kmeans(data(:, 1:3), Maximum);
    
    % QuestionH(c): Store final cluster centers and radii
    cluster_radii = zeros(Maximum, 1);
    for j = 1:Maximum
        cluster_points = data(cluster_idx == j, 1:3);
        distances = sqrt(sum((cluster_points - cluster_centers(j, :)).^2, 2));
        cluster_radii(j) = max(distances);  % The radius of cluster j
    end
    
    % Store the information for the current interval
    cls_cen{i} = struct('centers', cluster_centers, 'radii', cluster_radii);
end

% Question (i): Write a new data file containing each cluster center and its radius
fileID = fopen('data_saved.txt', 'w');
for i = 1:length(cls_cen)
    for j = 1:size(cls_cen{i}.centers, 1)
        fprintf(fileID, '%.4f %.4f %.4f %.4f\n', ...
            cls_cen{i}.centers(j, 1), cls_cen{i}.centers(j, 2), ...
            cls_cen{i}.centers(j, 3), cls_cen{i}.radii(j));
    end
end
fclose(fileID);

% Question (j): Display the new data file in Part (i) as a collection of transparent spheres
%Reference:https://www.mathworks.com/matlabcentral/answers/1733060-find-the-center-of-the-cluster-made-from-random-spheres-having-different-diameters
%ChatGpt
figure(2)
hold on;
for i = 1:length(cls_cen)
    for j = 1:size(cls_cen{i}.centers, 1)
        % Generate a sphere mesh
        [x, y, z] = sphere;
        % Scale and position the sphere mesh according to the cluster center and radius
        x = cls_cen{i}.radii(j) * x + cls_cen{i}.centers(j, 1);
        y = cls_cen{i}.radii(j) * y + cls_cen{i}.centers(j, 2);
        z = cls_cen{i}.radii(j) * z + cls_cen{i}.centers(j, 3);
        % Plot the sphere with red color and the specified transparency
        surf(x, y, z, 'FaceColor', 'red', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end
end


% Scatter plot for the transformed point cloud
scatter3(transformed_points(:, 1), transformed_points(:, 2), transformed_points(:, 3), ...
    10, [0 0 1], 'filled');

% Set viewing properties for better visualization
axis equal;  % Equal scaling for all axes


hold off;