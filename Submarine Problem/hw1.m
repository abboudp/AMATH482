clear all; close all; clc;

load subdata.mat

L = 10;
n = 64;

x2 = linspace(-L, L, n+1); 
x = x2(1:n); 
y = x; z = x;

k = (2*pi / (2*L)) * [0:(n / 2-1) - n / 2:-1];
ks = fftshift(k);

[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

a = zeros(n, n, n);

for j = 1:49
    Un(:,:,:) = reshape(subdata(:,j), n, n, n);
    a = a + fftn(Un);
end