clear;
% Add path
addpath('utils');
addpath('ESPIRiT_code');
addpath('coilCompression_code');
addpath('ismrm');
addpath('other code');
addpath('arrShow')
addpath('data');

% Wave Data
% load wave_Data.mat;
% load wave_ACS.mat;
% load wave_Py.mat;
% load wave_WavePy.mat;

% bssfp Data
load bssfp_Data.mat;
load bssfp_ACS.mat;
load bssfp_Py.mat;
load bssfp_WavePy.mat;
%% undersampled ===============================================
Ny = 203;
Ry = 4;
acs_len = 48;
%% -------------------------------------------------------------------------%
cali_idx = 1;
slice_idx = 1;
Data_Py = permute(squeeze(Data_Py(:,cali_idx,:,:)),[2,1,3]);
Data_WavePy = permute(squeeze(Data_WavePy(:,cali_idx,:,:)),[2,1,3]);
Data_ACS = permute(squeeze(Data_ACS(:,slice_idx,:,:)),[2,1,3]);
DATA_wave = permute(squeeze(DATA_wave(:,slice_idx,:,:,:)),[3,1,4,2]); % nFE*nPE*ncoil*nNum

Data_Py = rm_ROos(Data_Py, 1);
Data_WavePy = rm_ROos(Data_WavePy, 1);
Data_ACS = rm_ROos(Data_ACS, 1);
DATA_wave = rm_ROos(DATA_wave, 1);
%% Psf
Data_Py = raw_zeroPading_end(Data_Py,2,Ny);
Data_WavePy = raw_zeroPading_end(Data_WavePy,2,Ny);
Data_Py = fftshift(ifft(ifftshift(Data_Py,2),[],2),2);
Data_WavePy = fftshift(ifft(ifftshift(Data_WavePy,2),[],2),2);
psfy_raw = exp(1i * angle(squeeze(mean(Data_WavePy.*conj(Data_Py),3))));
PsfY = psf_fit2(psfy_raw);
% =========================================================================
% Data Prep
% =========================================================================
DATA_wave = raw_zeroPading_end(DATA_wave,2,Ny);
[Nx,Ny,Nc,Nnum] = size(DATA_wave);
Nx_os = size(PsfY,1);
num_maps = 2;
DATA = squeeze(Data_ACS(:,end-acs_len+1:end,:));
CalibSize = [acs_len, acs_len];
kCalib = crop(DATA,[CalibSize,Nc]);
% =========================================================================
% Coil Compression
% =========================================================================
nCHA_cc = 16; % compressed coil number
[sccmtx] = calcSCCMtx(kCalib);
sccmtx_cc = sccmtx(:,1:nCHA_cc);
kCalib = CC(kCalib,sccmtx_cc);
DATA = CC(DATA,sccmtx_cc);
DATA_wave_ccomp = zeros(Nx,Ny,nCHA_cc,Nnum);
for n = 1:Nnum
    DATA_wave_ccomp(:,:,:,n) = CC(DATA_wave(:,:,:,n),sccmtx_cc);
end
DATA_wave = DATA_wave_ccomp; clear DATA_wave_ccomp;
Nc = nCHA_cc;
% =========================================================================
% Extend along coil dimension
% =========================================================================
PsfY = repmat(PsfY,[1,1,Nc]);
mask_wave = double(abs(DATA_wave(:,:,1,1)) ~= 0);
samp_mat = repmat(mask_wave,[1,1,Nc]);
samp_mat_wave = repmat(mask_wave,[1,1,Nc]);
% =========================================================================
% Sensitivity Esimatioon
% =========================================================================
kSize = [8,8];
eigThresh_1 = 0.04;
eigThresh_2 = 0.87;
tic;
[K,S] = dat2Kernel(kCalib, kSize);
idx = find(S >= S(1)*eigThresh_1, 1, 'last');
[M,W] = kernelEig(K(:,:,:,1:idx), [Nx,Ny]);
toc;
% =========================================================================
% Reconstruction
% =========================================================================
Monte_num = 5;
res = zeros(Nx, Ny, Nnum, Monte_num);
org_DATA_wave = DATA_wave;
if num_maps == 1
    parpool(8);
    parfor gg = 1:Monte_num
        %for gg = 1:Monte_num
        noise = randn(size(org_DATA_wave))+1i*randn(size(org_DATA_wave));
        noise = bsxfun(@times,noise,samp_mat);
        % noise(:,:,:,6:8) = 0;
        DATA_wave = org_DATA_wave + noise;
        
        maps = M(:,:,:,end) .* repmat(W(:,:,end) > eigThresh_2, [1,1,Nc]);
        
        for n = 1:Nnum
            img_it_wave = ismrm_cartesian_iterative_Wave(squeeze(DATA_wave(:,:,:,n)), samp_mat, maps, PsfY);
            res(:,:,n,gg) = img_it_wave;
        end
    end
    delete(gcp);
elseif num_maps == 2
    parpool(8);
    parfor gg = 1:Monte_num
        % for gg = 1:Monte_num
        noise = randn(size(org_DATA_wave))+1i*randn(size(org_DATA_wave));
        noise = bsxfun(@times,noise,samp_mat);
        % noise(:,:,:,6:8) = 0;
        DATA_wave = org_DATA_wave + noise;
        
        maps = M(:,:,:,end-num_maps+1:end);
        weights = W(:,:,end-num_maps+1:end);
        weights = (weights - eigThresh_2)./(1-eigThresh_2).* (W(:,:,end-num_maps+1:end) > eigThresh_2);
        weights = -cos(pi*weights)/2 + 1/2;
        nIterCG = 50;
        
        WaveESP = WaveESPIRiT(maps, weights, [], PsfY);
        
        for n = 1:Nnum
            [reskWaveESPIRiT, resWaveESPIRiT] = cgESPIRiT(squeeze(DATA_wave(:,:,:,n)), WaveESP, nIterCG, 0.0, zeros(Nx,Ny,Nc));
            res(:,:,n,gg) = sum(resWaveESPIRiT,3);
        end
    end
    delete(gcp);
end
% res = res(49:208,49:208,:,:);
imshow(res(:,:,1),[]);

h1 = impoly;
Position_ROI1 = wait(h1);
mask1 = createMask(h1);

gfactor = squeeze(std(real(res), 0, 4))/sqrt(Ry);
as(gfactor)
figure;
% pcolor(gfactor);
pcolor(gfactor(:,:,1));
shading interp;
colorbar;

gfactor_wave = gfactor(:,:,1);
mean(gfactor_wave(mask1==1))



