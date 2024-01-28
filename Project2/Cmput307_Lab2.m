%Exercise1:
%Create box
%first row is horizontal coords;second row is vertical coords
my_pts = [2 2 3 3 2;2 3 3 2 2];

%write code here to display the original box

fig = figure(1);
figure(fig);

plot(my_pts(1,:),my_pts(2,:),'Color',[0 0 1]);
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
scatter(my_pts(1,:),my_pts(2,:));
scatter(my_rot_pts(1,:),my_rot_pts(2,:));
plot(my_rot_pts(1,:),my_rot_pts(2,:),'Color',[1 0 0]);
axis([1.5 4.5 0 3.5]);
hold off;

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Exercise2:
d_x = 1;
%howmuchever you want to translate in horizongtal direction
d_y = 3;
%howmuchever you want to translate in vertical direction

%create the origninal box
%first row has horizontal coords,second row has veritical coords
my_points = [1 1 2 2 1;1 2 2 1 1];

%write code to plot the original box

fig2 = figure(2);
figure(fig2);
scatter(my_points(1,:),my_points(2,:));
plot(my_points(1,:),my_points(2,:),'Color',[0 0 1]);
%write code to create Homo 2D Translattion Matrix my_trans using d_x and
%d_y
%add extra coord P =(px,py,ph),or x=(x,y,h),Catesian:x = (x/h,y/h),P' = T(tx,ty)P
my_trans = [1 0 d_x; 0 1 d_y; 0 0 1];

%Next, we perform the translation
%write code to convert my_points to the homogeneous system and store the
%result in hom_my_points, add '1' row
%since size return a mxn matrix size, m is # of row, n is # of cols
szdim2 = size(my_points,2);
one_row = ones(1,szdim2);
hom_my_points = [my_points(1,:);my_points(2,:);one_row];

%write code to perform translation in the homo system using my_trans
%and hom_my_points and store the result in trans_my_points

trans_my_points = my_trans*hom_my_points;
hold on;
scatter(my_points(1,:),my_points(2,:));
scatter(trans_my_points(1,:),trans_my_points(2,:));
plot(trans_my_points(1,:),trans_my_points(2,:),'Color',[1 0 0]);
%write code to plot the Translated box which has to be domne in Cartesian
%cut out the X,Y points and ignore the 3rd dim
axis([0.5 3.5 0.5 5.5]);
hold off;

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Exercise3:

d_y = 1; % howmuchever you want to translate in vertical direction
% create original box
% first row is horizontal and second row is vertical coordinates
my_pts = [3 3 4 4 3;3 4 4 3 3];
% write code to plot the box
fig1=figure(1);
% write code here to create your 2D rotation matrix my_rot
my_rot = [];
% write code to create your Homogeneous 2D Translation matrix hom_trans using d_x & d_y
hom_trans = [];
% Perform Compound transformation
% write code to construct your 2D Homogeneous Rotation Matrix using my_rot and store the
result in hom_rot
% HINT: start with a 3x3 identity matrix and replace a part of it with my_rot to create hom_rot
hom_rot = [];
% write code to convert my_pts to the homogeneous system and store the result in
%hom_my_pts
hom_my_pts = [];
% write code to perform in a single compound transformation: translation (hom_trans)
%followed by rotation (hom_rot) on hom_my_pts, and store the result in trans_my_pts
trans_my_pts = [];
% write code to plot the transformed box (output) which has to be done in Cartesian, so...
% cut out the X, Y points and ignore the 3rd dimension
hold on
axis([2 8 -2 5]); % just to make the plot nicely visible
% Now, let us reverse the order of rotation and translation and compare
fig2=figure(2);
plot(my_pts(1,1:end),my_pts(2,1:end),'b*-');
% write code to perform in a single compound transformation: rotation followed by translation,
%and store the result in trans_my_pts


