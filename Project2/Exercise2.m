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

fig1 = figure(1);
figure(fig1);
scatter(my_points(1,:),my_points(2,:));
plot(my_points(1,:),my_points(2,:),'b*-');
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
%scatter(my_points(1,:),my_points(2,:));
%scatter(trans_my_points(1,:),trans_my_points(2,:));
plot(trans_my_points(1,:),trans_my_points(2,:),'r*-');
%write code to plot the Translated box which has to be domne in Cartesian
%cut out the X,Y points and ignore the 3rd dim
axis([0.5 3.5 0.5 5.5]);
hold off;