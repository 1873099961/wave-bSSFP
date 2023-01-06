clear;
% Add path
addpath('utils');
addpath('ESPIRiT_code');
addpath('coilCompression_code');
addpath('ismrm');
addpath('ReadBayDataV5_2');
addpath('other code');
% genpath
%% retrosperspective undersampled ================================================
pathname = 'D:\MRI(only modify codes)\UIH_rawdata\2.bssfp gfactor\29.20220527_macaibing\';
file_Wave = [pathname, 'UID_132681783364326_bwave_T1R2dia_ref_1.raw'];

uih_prot = Read_UIH_Prot_fromRaw(file_Wave);
Ny = str2double(uih_prot.Root.Seq.KSpace.MatrixPE.Value.Text);
Ry = str2double(uih_prot.Root.Seq.PPA.PPAFactorPE.Value.Text);
acs_len = str2double(uih_prot.Root.Seq.PPA.RefLineLengthPE.Value.Text);


[DATA_wave,~,~,~,~,~,cTimes] = Read_UIH_Raw_v5_2(file_Wave);
[q,w,e,r] = size(DATA_wave); %WC sli=1 is different,reshape as 5D
DATA_wave = reshape(DATA_wave,q,1,w,e,r); %WC sli=1 is different,reshape as 5D
% [q,w,e] = size(DATA_wave); %WC sli=1 phs =1 is different,reshape as 5D
% DATA_wave = reshape(DATA_wave,q,1,1,w,e); %WC sli=1 phs =1 is different,reshape as 5D

DATA_wave = DATA_wave(25:129,:,:,:,:);
ppadata = DATA_wave(1:(size(DATA_wave,1)+acs_len)/2,:,:,:,:);  %  sli=1

% ppadata = DATA_wave(1:101,:,:,:,:); % WC:Asymmetric echo     sli=1
%% -------------------------------------------------------------------------%
cali_idx = 1;
slice_idx = 1;
Data_ACS = permute(squeeze(ppadata(:,slice_idx,1,:,:)), [2,1,3]);    % sli=1
DATA_wave = permute(squeeze(DATA_wave(:,slice_idx,:,:,:)),[3,1,4,2]); % nFE*nPE*ncoil*nNum
% DATA_wave = permute(squeeze(DATA_wave(:,slice_idx,:,:,:)),[2,1,3,4]); %WC  sli=1; phs=1;
DATA_wave = rm_ROos(DATA_wave, 1);
Data_ACS = rm_ROos(Data_ACS, 1);
% =========================================================================
% Data Prep
% =========================================================================
% DATA_wave = raw_zeroPading_end(DATA_wave,2,Ny);   % WC:if Asymmetric echo,no use
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
eigThresh_1 = 0.03;
eigThresh_2 = 0.85;

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
if num_maps == 1
    maps = M(:,:,:,end) .* repmat(W(:,:,end) > eigThresh_2, [1,1,Nc]);
    for n = 1:Nnum 
        img_it_sense(:,:,n) = ismrm_cartesian_iterative_SENSE_QZL(squeeze(DATA_wave(:,:,:,n)), samp_mat, maps);
        %img_it_wave(:,:,n) = ismrm_cartesian_iterative_Wave(squeeze(DATA_wave(:,:,:,n)), samp_mat, maps, PsfY);
    end
elseif num_maps == 2
    maps = M(:,:,:,end-num_maps+1:end);
    
    weights = W(:,:,end-num_maps+1:end);
    weights = (weights - eigThresh_2)./(1-eigThresh_2).* (W(:,:,end-num_maps+1:end) > eigThresh_2);
    weights = -cos(pi*weights)/2 + 1/2;
    %nIterCG = 100;
    nIterCG = 50;
    
    ESP = ESPIRiT(maps, weights);
%    WaveESP = WaveESPIRiT(maps, weights, [], PsfY);
    for n = 1:Nnum
        [reskESPIRiT, resESPIRiT] = cgESPIRiT(squeeze(DATA_wave(:,:,:,n)), ESP, nIterCG, 0.0, zeros(Nx,Ny,Nc));
        %[reskWaveESPIRiT, resWaveESPIRiT] = cgESPIRiT(squeeze(DATA_wave(:,:,:,n)), WaveESP, nIterCG, 0.0, zeros(Nx,Ny,Nc));
        img_it_sense(:,:,n) = sum(resESPIRiT,3);
    end

end

% T1 mapping can not normlization
% img_it_sense = img_it_sense/max(max(max(abs(ifft2c(img_it_sense)))));
as(img_it_sense)
%% Asymmetric echo
bb = fft2c(img_it_sense);
aa = zeros(192,153,Nnum);
aa(:,25:129,:) = bb;
ww = ifft2c(aa);
% as(ww)
%%%%% Registeration %%%%%----------------------------------------------------------------
[RO,PE,PHS] = size(ww);
Reg = zeros(RO,PE,PHS);
moving = abs(ww);
fixed = abs(ww(:,:,1));
for i = 1:5
    [~,movingReg] = imregdemons(moving(:,:,i),fixed,'AccumulatedFieldSmoothing',2.5); % 0.5-3.0; maybe 1.5-2.0
    Reg(:,:,i) = movingReg;
end
for i = 9:11
    [~,movingReg] = imregdemons(moving(:,:,i),fixed,'AccumulatedFieldSmoothing',2.5); % 0.5-3.0; maybe 1.5-2.0
    Reg(:,:,i) = movingReg;
end

TI_array = uih_prot.Root.Seq.Basic.TI.Value;
TI = [str2double(TI_array{1}.Text), str2double(TI_array{2}.Text)]/1000; % ms
RRs = [cTimes(1:5) - cTimes(1); cTimes(9:11) - cTimes(9)];
TIs = [TI(1)+RRs(1:5); TI(2)+RRs(6:8)]; 
data = abs(cat(3, Reg(:,:,1:5), Reg(:,:,9:11)));
[TIs, idx] = sort(TIs,'ascend');
data = data(:,:,idx);

%%----fitting------------
data1D = reshape(data, [], size(data, 3))';
timingPerPoint = TIs*ones([1,size(data,1)*size(data,2)]);    
[A, B, T1star] = molli_fit(data, TIs');
T1 = T1star.*((B./A) - 1);
T1(A == 0) = 0;

tmp = medfilt2(T1, [3, 3]);
mask = (B./A) < 1.00001;
T1(mask) = tmp(mask);

imshow(fixed,[]);
h1 = impoly;
Position_ROI1 = wait(h1);
mask1 = createMask(h1);
T1(mask1) = tmp(mask1);

figure;imagesc(flipud(T1'), [0 2000]); axis off;colormap('jet');colorbar;
% figure;imagesc(flipud(T1'), [0 2000]); axis equal;colormap('jet');colorbar;

% save([file_Wave, '_res.mat'], 'img_it_wave_2map', 'T1', 'uih_prot');






