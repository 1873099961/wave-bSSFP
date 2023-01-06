% load ref,maybe dim is ro,pe phs
% load recon,maybe dim is ro,pe,phs
clear;
load full_none.mat
ref = img_it_sense(:,:,1);
load R4_none.mat
recon = img_it_sense(:,:,1);
load mask1.mat
ref = ref/max(max(max(abs(ifft2c(ref)))));
recon = recon/max(max(max(abs(ifft2c(recon)))));

% figure;
% imshow(ref,[])
% h1 = impoly;
% Position_ROI1 = wait(h1);
% mask1 = createMask(h1);

ref = proc_norm(ref, mask1);
recon = proc_norm(recon, mask1);
mse = abs(ref - recon);

figure;
pcolor(abs(mse));
shading interp;
colorbar;
caxis([0,0.1]);

img_it_sense_rmse = sqrt(sum(sum(abs(mse).^2,1),2)) / sqrt(sum(sum(abs(ref).^2,1),2))
% caxis([0,0.05]);
% can set scan,use caxis([m,n]);
