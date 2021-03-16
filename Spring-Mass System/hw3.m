clear all; close all; clc

test = 1;

load(['./cam1_' num2str(test) '.mat']);
load(['./cam2_' num2str(test) '.mat']);
load(['./cam3_' num2str(test) '.mat']);

if test == 1
    view1 = vidFrames1_1;
    view2 = vidFrames2_1;
    view3 = vidFrames3_1;
elseif test == 2
    view1 = vidFrames1_2;
    view2 = vidFrames2_2;
    view3 = vidFrames3_2;
elseif test == 3
    view1 = vidFrames1_3;
    view2 = vidFrames2_3;
    view3 = vidFrames3_3;
else
    view1 = vidFrames1_4;
    view2 = vidFrames2_4;
    view3 = vidFrames3_4;
end

[s1a,s1b,s1c,s1d] = size(view1);
[s2a,s2b,s2c,s2d] = size(view2);
[s3a,s3b,s3c,s3d] = size(view3);

area = 18;
Coord = [322,289];                
X1 = zeros(1,s1d);
Y1 = zeros(1,s1d);

for i=1:1:s1d
    img = rgb2gray(view1(:,:,:,i));
    img(:,1:Coord(1)-area) = 0;
    img(:,Coord(1)+area:end) = 0;
    img(1:Coord(2)-area,:) = 0;
    img(Coord(2)+area:end,:) = 0;
    [Z,index] = max(img(:));
    [P1,P2] = ind2sub(size(img),index);
    X1(i) = P2;
    Y1(i) = P1;
    Coord = [P2, P1];
end

Coord = [239,294];                
X2 = zeros(1,s2d);
Y2 = zeros(1,s2d);
for i=1:1:s2d
    img = rgb2gray(view2(:,:,:,i));
    img(:,1:Coord(1)-area) = 0;
    img(:,Coord(1)+area:end) = 0;
    img(1:Coord(2)-area,:) = 0;
    img(Coord(2)+area:end,:) = 0;
    [Z,index] = max(img(:));
    [P1,P2] = ind2sub(size(img),index);
    X2(i) = P2;
    Y2(i) = P1;
    Coord = [P2, P1];
end

Coord = [355,234];                
X3 = zeros(1,s3d);
Y3 = zeros(1,s3d);

for i=1:1:s3d
    img = rgb2gray(view3(:,:,:,i));
    img(:,1:Coord(1)-area) = 0;
    img(:,Coord(1)+area:end) = 0;
    img(1:Coord(2)-area,:) = 0;
    img(Coord(2)+area:end,:) = 0;
    [Z,index] = max(img(:));
    [P1,P2] = ind2sub(size(img),index);
    X3(i) = P1;
    Y3(i) = P2;
    Coord = [P2, P1];
end

N = min(min(s1d,s2d),s3d);
X = [X1(1:N);Y1(1:N);X2(1:N);Y2(1:N);X3(1:N);Y3(1:N)];

[M,N] = size(X);
mean = mean(X,2);
X = X - repmat(mean,1,N);
[U, S, V] = svd(X'/sqrt(N-1));
lam = diag(S).^2; 

Y = V' * X;

figure(1)
hold on
plot(Y(1,:))
plot(Y(2,:))

legend('Component 1', 'Component 2','Orientation','horizontal','Location','south')
xlabel('Frame')
ylabel('Position')
title('Position from Principal Components')

figure(2)
plot(1:6, lam/sum(lam), 'o', 'Linewidth', 1);
title("Case 1 Energy Captured by Rank");
xlabel("Rank"); 
ylabel("Energy Captured by Component");