%Question(A):
X = pcread('dragondata/dragon.ply');
figure(1)
hold on;
scatter3(X.Location(:,1), X.Location(:,2), X.Location(:,3),"b.");

%  ---------------- Write your code here  ---------------------------


% Release the hold on the current figure
hold off;

%Questioin(B)
% Determine the number of points in the point cloud
N = size(X.Location, 1);

% Display the number of points in the command window
disp(['Number of points in the model: ' num2str(N)]);

%Number of points in the model: 437645

%  ------------------------------------------------------------------

%Question(C):
% Perform PCA on the point cloud data
coeff = pca(X.Location);

% The first eigenvector
firstEigenvector = coeff(:, 1);

% Display the first eigenvector in the command window
disp('The first eigenvector of the PCA:');
disp(firstEigenvector);

%The first eigenvector of the PCA:
        % 0.8899
        %-0.4554
        % 0.0252

%  ------------------------------------------------------------------
%Question(D):
% The first eigenvector
firstEigenvector = coeff(:, 1);

% Normalized first eigenvector
v = firstEigenvector / norm(firstEigenvector);

% The desired new x-axis
x_axis = [1; 0; 0];

% Compute the axis of rotation (cross product of first eigenvector and x-axis)
axis_of_rotation = cross(v, x_axis);

% Compute the angle of rotation (using the dot product and inverse cosine)
angle_of_rotation = acos(dot(v, x_axis));

% Normalize the axis of rotation
axis_of_rotation = axis_of_rotation / norm(axis_of_rotation);

% Here you would replace vrrotvec2mat with the method provided in your notes
% For demonstration, this line is kept as is:
R = vrrotvec2mat([axis_of_rotation; angle_of_rotation]);

% Transform the point cloud
transformed_points = (R * X.Location')';

%  ------------------------------------------------------------------
%Question(E):
% Sort the transformed points following Part (d), based on the new X values.
[~, sortIdx] = sort(transformed_points(:, 1), 'ascend');
sortedTransformedPoints = transformed_points(sortIdx, :);

%  ------------------------------------------------------------------
%Question(F):
% Determine the Minimum X value MinX, and Maximum X value MaxX, following Part (e).
MinX = min(sortedTransformedPoints(:, 1));
MaxX = max(sortedTransformedPoints(:, 1));

% Display the minimum and maximum X values in the command window
disp(['Minimum X value: ' num2str(MinX)]);
disp(['Maximum X value: ' num2str(MaxX)]);

%Minimum X value: -0.169
%Maximum X value: 0.039677
%  ------------------------------------------------------------------


% Question(G): Divide interval into 100 parts with approximately the same number of points
interval_points = linspace(MinX, MaxX, 101);
clusters_info = cell(100, 1);  % Cell array to hold cluster info for each interval

for i = 1:100
    % Get the points in the current interval
    in_interval = transformed_points(:, 1) >= interval_points(i) & transformed_points(:, 1) < interval_points(i+1);
    interval_data = transformed_points(in_interval, :);
    
    % QuestionH(a): Create N/1000 clusters
    k = max(round(N/1000), 1);  % Ensure at least one cluster
    if size(interval_data, 1) < k
        k = size(interval_data, 1);  % Adjust if fewer points than clusters
    end
    
    % QuestionH(b): Perform k-means clustering
    opts = statset('Display','final');
    [cluster_idx, cluster_centers] = kmeans(interval_data(:, 1:3), k, 'Options', opts, 'MaxIter', 1000, 'Replicates', 5);
    
    % QuestionH(c): Store final cluster centers and radii
    cluster_radii = zeros(k, 1);
    for j = 1:k
        cluster_points = interval_data(cluster_idx == j, 1:3);
        distances = sqrt(sum((cluster_points - cluster_centers(j, :)).^2, 2));
        cluster_radii(j) = max(distances);  % The radius of cluster j
    end
    
    % Store the information for the current interval
    clusters_info{i} = struct('centers', cluster_centers, 'radii', cluster_radii);
end

% Question (i): Write a new data file containing each cluster center and its radius
fileID = fopen('cluster_centers_and_radii.txt', 'w');
for i = 1:length(clusters_info)
    for j = 1:size(clusters_info{i}.centers, 1)
        fprintf(fileID, '%.4f %.4f %.4f %.4f\n', ...
            clusters_info{i}.centers(j, 1), clusters_info{i}.centers(j, 2), ...
            clusters_info{i}.centers(j, 3), clusters_info{i}.radii(j));
    end
end
fclose(fileID);

% Question (j): Display the new data file in Part (i) as a collection of transparent spheres
figure(2)
hold on;
for i = 1:length(clusters_info)
    for j = 1:size(clusters_info{i}.centers, 1)
        % Generate a sphere mesh
        [x, y, z] = sphere;
        % Scale and position the sphere mesh according to the cluster center and radius
        x = clusters_info{i}.radii(j) * x + clusters_info{i}.centers(j, 1);
        y = clusters_info{i}.radii(j) * y + clusters_info{i}.centers(j, 2);
        z = clusters_info{i}.radii(j) * z + clusters_info{i}.centers(j, 3);
        % Plot the sphere with red color and the specified transparency
        surf(x, y, z, 'FaceColor', 'red', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end
end

% Scatter plot for the transformed point cloud
scatter3(transformed_points(:, 1), transformed_points(:, 2), transformed_points(:, 3), ...
    10, [0 0 1], 'filled');

% Set viewing properties for better visualization
axis equal;  % Equal scaling for all axes

%camlight headlight; % Add a light at the camera position
%lighting gouraud; % Gouraud lighting to smooth the appearance of the spheres
hold off;