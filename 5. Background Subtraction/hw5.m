close all; clear all; clc

% frames = VideoReader('monte_carlo_low.mp4');
frames = VideoReader('ski_drop_low.mp4');
vidHeight = frames.Height;
vidWidth = frames.Width;
totalFrames = frames.NumberOfFrames;

X = zeros(vidWidth*vidHeight, totalFrames);
for i = 1:1:totalFrames
    current_Frame2 = rgb2gray(read(frames,i));
    current_Frame = reshape(current_Frame2,vidWidth*vidHeight,1);
    X(:,i) = double(current_Frame);
end

t_current = linspace(0, frames.CurrentTime, totalFrames+1);
t_frame = t_current(1:end-1);
dt = t_frame(2) - t_frame(1);

X1 = X(:,1:end-1);
X2 = X(:,2:end);
[U,Sigma,V] = svd(X1, 'econ');

figure(1)
plot(diag(Sigma)/sum(diag(Sigma)), '-o')
xlabel('Rank','Fontsize',12)
ylabel('Proportion of Energy Captured','Fontsize',12)

energy = 0;
total = sum(diag(Sigma));
threshold = 0.9; 
r = 0;
while energy < threshold
    r = r + 1;
    energy = energy + Sigma(r,r)/total;
end

U2 = U(:,1:r);
S2 = Sigma(1:r,1:r);
V2 = V(:,1:r);
S = U2'*X2*V2/S2;
[E,D] = eig(S);
phi = U2*E;

mu = diag(D);
w = log(mu)/dt;
b = find(abs(w) < 1e-2);
w_b = w(b);
phi_b = phi(:,b);

X1_n = X1(:,1);
y = phi_b \ X1_n;
M = length(t_frame);
modes = zeros(length(w_b), M);
for j = 1:M
    modes(:,j) = (y.*exp(w_b*t_frame(j)));
end

X_low = phi_b*modes;
% X_sparse = X - abs(X_low);
% R = X_sparse.*(X_sparse < 0);
% X_low = R + abs(X_low);
% X_sparse = X_sparse - R;
X_low = abs(X_low);
X_sparse = X - abs(X_low) + 200;

% capture = [100, 200];
capture = [200, 400];
figure(2)
subplot(3,1,1)
fig1 = reshape(uint8(X(:,capture(1))), vidHeight, vidWidth);
imshow(fig1)
title(['Frame ', num2str(capture(1)),' of Original'])
subplot(3,1,2)
fig2 = reshape(uint8(X_low(:,capture(1))), vidHeight, vidWidth);
imshow(fig2)
title(['Frame ', num2str(capture(1)),' of Background'])
subplot(3,1,3)
fig3 = reshape(uint8(X_sparse(:,capture(1))), vidHeight, vidWidth);
imshow(fig3)
title(['Frame ', num2str(capture(1)),' of Foreground'])

figure(3)
subplot(3,1,1)
fig1 = reshape(uint8(X(:,capture(2))), vidHeight, vidWidth);
imshow(fig1)
title(['Frame ', num2str(capture(2)),' of Original'])
subplot(3,1,2)
fig2 = reshape(uint8(X_low(:,capture(2))), vidHeight, vidWidth);
imshow(fig2)
title(['Frame ', num2str(capture(2)),' of Background'])
subplot(3,1,3)
fig3 = reshape(uint8(X_sparse(:,capture(2))), vidHeight, vidWidth);
imshow(fig3)
title(['Frame ', num2str(capture(2)),' of Foreground'])
