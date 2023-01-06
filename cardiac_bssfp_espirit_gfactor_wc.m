clear;
% Add path
addpath('utils');
addpath('ESPIRiT_code');
addpath('coilCompression_code');
addpath('ismrm');
% addpath('ReadBayData_wave')
addpath('ReadBayDataV5_2');
addpath('other code');
% genpath
% load mask1(32).mat;
% load mask2(24).mat;
% load cardiacmask3(24).mat;

%% retrosperspective undersampled ================================================
% pathname = '/home/zhilangqiu/WC/bssfp_wave_2/';
% file_Wave = [pathname, 'UID_248590042715384_gre_bssfp_wave2d_none_Amp20_full.raw'];
% pathname = 'D:\MRI（only modify codes）\CSandAI cine rawdata\2.bssfp gfactor\3(coils24)\';
% file_Wave = [pathname, 'UID_103102321475193_gre_bssfp_wave2d_None_R4.raw'];
pathname = 'D:\MRI(only modify codes)\UIH_rawdata\2.bssfp gfactor\19.20220420_tanyuewu\g\';
file_Wave = [pathname, 'UID_32239674962473_bwave_locR4sys_amp0_4ch.raw'];

uih_prot = Read_UIH_Prot_fromRaw(file_Wave);
Ny = str2double(uih_prot.Root.Seq.KSpace.MatrixPE.Value.Text);
Ry = str2double(uih_prot.Root.Seq.PPA.PPAFactorPE.Value.Text);
acs_len = str2double(uih_prot.Root.Seq.PPA.RefLineLengthPE.Value.Text);


[DATA_wave] = Read_UIH_Raw_v5_2(file_Wave);
% [q,w,e,r] = size(DATA_wave); %WC sli=1 is different,reshape as 5D
% DATA_wave = reshape(DATA_wave,q,1,w,e,r); %WC sli=1 is different,reshape as 5D
[q,w,e] = size(DATA_wave); %WC sli=1 phs =1 is different,reshape as 5D
DATA_wave = reshape(DATA_wave,q,1,1,w,e); %WC sli=1 phs =1 is different,reshape as 5D
% DATA_wave = DATA_wave(:,1,1,:,:);
DATA_wave = DATA_wave(25:129,:,:,:,:);

ppadata = DATA_wave(1:(size(DATA_wave,1)+acs_len)/2,:,:,:,:); 
% ppadata = DATA_wave(1:101,:,:,:,:); 
%% -------------------------------------------------------------------------%
cali_idx = 1;
slice_idx = 1;
Data_ACS = permute(squeeze(ppadata(:,slice_idx,1,:,:)), [2,1,3]);
% DATA_wave = permute(squeeze(DATA_wave(:,slice_idx,:,:,:)),[3,1,4,2]); % nFE*nPE*ncoil*nNum
DATA_wave = permute(squeeze(DATA_wave(:,slice_idx,:,:,:)),[2,1,3,4]); %WC  sli=1; phs=1;
DATA_wave = rm_ROos(DATA_wave, 1);
Data_ACS = rm_ROos(Data_ACS, 1);
% =========================================================================
% Data Prep
% =========================================================================
DATA_wave = raw_zeroPading_end(DATA_wave,2,Ny);   
[Nx,Ny,Nc,Nnum] = size(DATA_wave);


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
DATA_wave = DATA_wave_ccomp;
DATA_ref = DATA_wave_ccomp;
clear DATA_wave_ccomp;
Nc = nCHA_cc;

% =========================================================================
% Extend along coil dimension
% =========================================================================
mask_wave = double(abs(DATA_wave(:,:,1,1)) ~= 0);
samp_mat = repmat(mask_wave,[1,1,Nc]);           % Rfactor/full use this samp_mat
samp_mat_wave = repmat(mask_wave,[1,1,Nc]);      % Rfactor/full use this samp_mat
                            % use Rfactor data samp_mat is DATA_wave+Data_ACS
                            
% mask = zeros(Nx,Ny);
% mask(:,floor(Ny/2)+1:-Ry:1) = 1; 
% mask(:,floor(Ny/2)+1:Ry:end) = 1;
% DATA_wave = bsxfun(@times, DATA_wave, mask);     % mask Rfactor
% mask_wave = mask;                                  % Rfactor use this samp_mat
% samp_mat = repmat(mask_wave,[1,1,Nc]);             % Rfactor use this samp_mat
%                             % use Rfactor data samp_mat is only maskRfactor
% samp_mat_wave = repmat(mask_wave,[1,1,Nc]);

% =========================================================================
% Sensitivity Esimatioo1n
% =========================================================================
 kSize = [8,8];
eigThresh_1 = 0.025;
eigThresh_2 = 0.8;

tic;
[K,S] = dat2Kernel(kCalib, kSize);
idx = find(S >= S(1)*eigThresh_1, 1, 'last');
[M,W] = kernelEig(K(:,:,:,1:idx), [Nx,Ny]);
toc;

% as(W)
% maps = M(:,:,:,end) .* repmat(W(:,:,end) > eigThresh_2,[1,1,Nc]);as(maps)

%Nnum = 1;
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
            img_it_wave = ismrm_cartesian_iterative_SENSE_QZL(squeeze(DATA_wave(:,:,:,n)), samp_mat, maps);
            res(:,:,n,gg) = img_it_wave;
        end
    end
    delete(gcp);
elseif num_maps == 2
    parpool(8);
    parfor gg = 1:Monte_num
    %for gg = 1:Monte_num
        noise = randn(size(org_DATA_wave))+1i*randn(size(org_DATA_wave));
        noise = bsxfun(@times,noise,samp_mat);
        % noise(:,:,:,6:8) = 0;
        DATA_wave = org_DATA_wave + noise;
        
        maps = M(:,:,:,end-num_maps+1:end);
        weights = W(:,:,end-num_maps+1:end);
        weights = (weights - eigThresh_2)./(1-eigThresh_2).* (W(:,:,end-num_maps+1:end) > eigThresh_2);
        weights = -cos(pi*weights)/2 + 1/2;
        nIterCG = 50;
        
        ESP = ESPIRiT(maps, weights);
        % WaveESP = WaveESPIRiT(maps, weights, [], PsfY);
        
        for n = 1:Nnum
            [reskWaveESPIRiT, resWaveESPIRiT] = cgESPIRiT(squeeze(DATA_wave(:,:,:,n)), ESP, nIterCG, 0.0, zeros(Nx,Ny,Nc));
            res(:,:,n,gg) = sum(resWaveESPIRiT,3);
        end
    end
    delete(gcp);
end

% load mask1.mat
% res = res(49:208,49:208,:,:);
imshow(res(:,:,1),[]);

h1 = impoly;
Position_ROI1 = wait(h1);
mask1 = createMask(h1);

gfactor = squeeze(std(real(res), 0, 4))/sqrt(Ry);
as(gfactor)
% g=zeros(160,160);
% g(mask1) =gfactor(mask1);
% as(g)
% figure;
% % pcolor(g);
% pcolor(gfactor(:,:,1));
% % caxis([0,1]);
% shading interp;
% colorbar;

gfactor_wave = gfactor(:,:,1);
mean(gfactor_wave(mask1==1))