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
%write code to create Homo 2D Translattion Matrix my_trans using d_x and
%d_y
%P =(px,py,ph),x=(x,y,h),Catesian:x = (x/h,y/h),P' = T(tx,ty)P

my_trans =[];
%Next, we perform the translation

%write code to convert my_points to the homogeneous system using my_trans
%and hom_my_points and store the result in trans_my_points

trans_my_points = [];
hold on;
%write code to plot the Translated box which has to be domne in Cartesian
%cut out the X,Y points and ignore the 3rd dim
axis([0.5 3.5 0.5 5.5]);





