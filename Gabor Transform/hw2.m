%% GNR Guitar
close all; clear all; clc;
[y, Fs] = audioread('GNR.m4a');
v = y';

L = length(v)/Fs;
n = length(v);
t2 = linspace(0,L,n+1); 
t = t2(1:n); 
k = (1/L)*[0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

tau = 0:0.2:L;
a = 100;
Vgt_spec=[];

for j = 1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Vg = g.*v;
    Vgt = fft(Vg);
    [M, I] = max(fftshift(abs(Vgt)));
    notes(j) = abs(ks(I));
    Vgt_spec(:,j)= abs(fftshift(Vgt));
end

figure(1);
pcolor(tau, ks, Vgt_spec), shading interp 
set(gca,'ylim',[300 800],'Fontsize',16) 
xlabel(['Time (s)'])
ylabel(['Frequency (Hz)'])
title('Spectrogram for Guitar in GNR');
colormap(hot)
figure(2);
plot(tau, notes, 'k.','Markersize',10);
yticks([311.1, 370, 415.3, 554.4,698.5, 740]);
yticklabels({'D3#','F4#','G4#','C4#','F4','F5#'});
ylim([300 800])
ylabel('Notes')
yyaxis right
ylabel('Frequencies (Hz)')
ylim([300 800])
title('Score for Guitar in GNR');
xlabel('Time (s)'); 

%% Floyd Bass Unfiltered
close all; clear all; clc;
[y, Fs] = audioread('Floyd.m4a');
v = y';

L = length(v)/Fs;
n = length(v);
t2 = linspace(0,L,n+1); 
t = t2(1:n); 
k = (1/L)*[0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

t = t(1:n-1);
v = v(1:n-1);

tau = 0:2:L;
a = 100;
a2 = 0.2;
Vgt_spec=[];

for j = 1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Vg = g.*v;
    Vgt = fft(Vg);
    [M, I] = max(fftshift(abs(Vgt)));
    notes(j) = abs(ks(I));
    Vgt_spec(:,j)= abs(fftshift(Vgt));
end

figure(3);
pcolor(tau, ks, Vgt_spec), shading interp 
set(gca,'ylim',[0 300],'Fontsize',16) 
xlabel(['Time (s)'])
ylabel(['Frequency (Hz)'])
title("Spectrogram for Bass in Comfortably Numb");
colormap(hot)
figure(4);
plot(tau, notes, 'k.','Markersize',10);
yticks([82.41, 98, 110, 123.5, 164.8, 246.9]);
yticklabels({'E','G2','A2','B2','E2','B3'});
ylabel('Notes')
ylim([0 300])
yyaxis right
ylabel('Frequencies (Hz)')
ylim([0 300])
title('Score for Bass in Comfortably Numb');
xlabel('Time (s)'); 

%% Floyd Bass Filtered
close all; clear all; clc;
[y, Fs] = audioread('Floyd.m4a');
v = y';

L = length(v)/Fs;
n = length(v);
t2 = linspace(0,L,n+1); 
t = t2(1:n); 
k = (1/L)*[0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

tau = 0:2:L;
a = 100;
a2 = 0.2;
Vgt_spec=[];

t = t(1:n-1);
v = v(1:n-1);

for j = 1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Vg = g.*v;
    Vgt = fft(Vg);
    [M, I] = max(fftshift(abs(Vgt)));
    notes(j) = abs(ks(I));
    g2 = exp(-a2 * (ks-notes(j)).^2);
    VgtF = Vgt.*fftshift(g2);
    Vgt_spec(:,j)= abs(fftshift(VgtF));
end

figure(5);
pcolor(tau, ks, Vgt_spec), shading interp 
set(gca,'ylim',[0 300],'Fontsize',16) 
xlabel(['Time (s)'])
ylabel(['Frequency (Hz)'])
title("Spectrogram for Bass in Comfortably Numb");
colormap(hot)
figure(6);
plot(tau, notes, 'k.','Markersize',10);
yticks([82.41, 98, 110, 123.5, 164.8, 246.9]);
yticklabels({'E','G2','A2','B2','E2','B3'});
ylabel('Notes')
ylim([0 300])
yyaxis right
ylabel('Frequencies (Hz)')
ylim([0 300])
title('Score for Bass in Comfortably Numb');
xlabel('Time (s)'); 

%% Floyd Guitar Filtered

close all; clear all; clc;
[y, Fs] = audioread('Floyd.m4a');
v = y';

L = length(v)/Fs;
n = length(v);
t2 = linspace(0,L,n+1); 
t = t2(1:n); 
k = (1/L)*[0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

vG = fftshift(fft(v));
for j = 1:length(ks)
    if abs(ks(j)) < 260
        vG(j) = 0;
    end
end
v = ifft(ifftshift(vG));

t = t(1:n-1);
v = v(1:n-1);

tau = 0:2:L;
a = 100;
a2 = 0.2;
Vgt_spec=[];

for j = 1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Vg = g.*v;
    Vgt = fft(Vg);
    [M, I] = max(fftshift(abs(Vgt)));
    notes(j) = abs(ks(I));
    g2 = exp(-a2 * (ks-notes(j)).^2);
    VgtF = Vgt.*fftshift(g2);
    Vgt_spec(:,j)= abs(fftshift(VgtF));
end

figure(7);
pcolor(tau, ks, Vgt_spec), shading interp 
set(gca,'ylim',[300 1000],'Fontsize',16) 
xlabel(['Time (s)'])
ylabel(['Frequency (Hz)'])
title('Spectrogram for Guitar in Comfortably Numb');
colormap(hot)
figure(8);
plot(tau, notes, 'k.','Markersize',10);
yticks([293.7, 329.6, 370, 440, 493.9, 587.3, 659.3, 740, 830.6, 880, 987.8]);
yticklabels({'D3','E3','F4#','A4','B4','D4','E4','F5#','G5#','A5','B5'});
ylim([300 1000])
ylabel('Notes')
yyaxis right
ylabel('Frequencies (Hz)')
ylim([300 1000])
title('Score for Guitar in Comfortably Numb');
xlabel('Time (s)'); 