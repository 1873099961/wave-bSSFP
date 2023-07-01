function [A, B, T1star] = molli_fit(img, t)
img = 3000.*img./max(abs(img(:)));
[npe, nfe, nphase] = size(img);
ref = mean(abs(img), 3);
ref_gray = mat2gray(ref); 
threshold = graythresh(ref_gray);
background_level = threshold*max(ref(:));   
mask = ref > background_level*0.1;

A   = zeros(npe, nfe);
B    = zeros(npe, nfe);
T1star = zeros(npe, nfe);
lb = [0, 2, 1];
ub = [5000, 20, 3000];
options = optimset('lsqnonlin');
bar=waitbar(0,'Please wait...');

%parpool(5);
%parfor h = 1:npe
for h = 1:npe
    for w = 1:nfe
        if mask(h,w) ~= 0
            ydata = squeeze(img(h, w,:))';
            x0 = [mean(ydata(end-2:end)); 2*mean(ydata(end-2:end)); 900];
            [x, resnorm] = lsqcurvefit(@molli, x0, t, ydata , [], [], options);
            A(h,w) = x(1);
            B (h,w) = x(2);
            T1star(h,w) = x(3);
        end
        waitbar( ((h-1)*nfe+w)/(nfe * npe),bar);
    end
end
delete(bar); % remove timer bar
%delete(gcp);

function F = molli(x, xdata)
%F = x(1) - x(2)*exp(-xdata./x(3));
F = abs(x(1) - x(2)*exp(-xdata./x(3)));


