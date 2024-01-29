%Exercise1:
%Create box
%first row is horizontal coords;second row is vertical coords
my_pts = [2 2 3 3 2;2 3 3 2 2];

%write code here to display the original box

fig = figure(1);
figure(fig);

plot(my_pts(1,:),my_pts(2,:),'b*-');
%xlim([1.5 4.5]);
%ylim([0 3.5]);

%write code here to create your 2D rotation matrix my_rot,suggest degree 30
%Do the dot product:P' = RP
a = -pi/6;
my_rot=[cos(a) -sin(a);sin(a) cos(a)];

%write code to perform rotation using my_rot and my_pts and store the
%result in my_rot_pts
my_rot_pts = my_rot*my_pts;

%Use hold on/ hold off ,plot multiple sets of data on the same figure
% without erasing the existing data
hold on;
%write code to plot output
%scatter(my_pts(1,:),my_pts(2,:));
%scatter(my_rot_pts(1,:),my_rot_pts(2,:));
plot(my_rot_pts(1,:),my_rot_pts(2,:),'r*-');
axis([1.5 4.5 0 3.5]);
hold off;