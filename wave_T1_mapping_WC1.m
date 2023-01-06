clear all; close all;
% Add path
addpath('utils');
addpath('ESPIRiT_code');
addpath('coilCompression_code');
addpath('ismrm');
addpath('ReadBayData_wave')
addpath('other code');
addpath('nufft_files');

pathname = 'D:\MRI(only modify codes)\UIH_rawdata\2.bssfp gfactor\29.20220527_macaibing\';
file_Wave = [pathname, 'UID_132681783364297_bwave_T1R4dia_amp14_1.raw'];

[DATA_wave,ppadata,phasecor,feed_back,noise_scan,maxCh,Data_ACS,Data_Py,Data_WavePy, cTimes] = Read_UIH_Raw_wave(file_Wave);

%% for single image
if ndims(DATA_wave) == 3
    DATA_wave = permute(DATA_wave, [1, 4, 2, 3]);
end

%%
Data_Py = permute(Data_Py,[2,1,3]);  
Data_WavePy = permute(Data_WavePy,[2,1,3]);
Data_ACS = permute(Data_ACS,[2,1,3]); 
DATA_wave = permute(DATA_wave,[3,1,4,2]); % nFE*nPE*ncoil*nNum

Data_Py = rm_ROos(Data_Py, 1);
Data_WavePy = rm_ROos(Data_WavePy, 1);
Data_ACS = rm_ROos(Data_ACS, 1);
DATA_wave = rm_ROos(DATA_wave, 1);
% =========================================================================
% Read Recon paras and calcualte Psf
% =========================================================================
uih_prot = Read_UIH_Prot_fromRaw(file_Wave);
Ny = str2double(uih_prot.Root.Seq.KSpace.MatrixPE.Value.Text);
Ry = str2double(uih_prot.Root.Seq.PPA.PPAFactorPE.Value.Text); 
acs_len = str2double(uih_prot.Root.Seq.PPA.RefLineLengthPE.Value.Text);

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

useVCC = 0;
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
% Virtual Conjugate Coil	
% =========================================================================
if useVCC == 1
    v_kCalib = conj(flip(flip(kCalib,1),2)); 
    if mod(size(v_kCalib,1),2) == 0
        v_kCalib = circshift(v_kCalib,[1,0]);
    end
    if mod(size(v_kCalib,2),2) == 0
        v_kCalib = circshift(v_kCalib,[0,1]);
    end
    kCalib = cat(3, kCalib, v_kCalib);
    
    v_DATA = conj(flip(flip(DATA,1),2)); 
    if mod(size(v_DATA,1),2) == 0
        v_DATA = circshift(v_DATA,[1,0]);
    end
    if mod(size(v_DATA,2),2) == 0
        v_DATA = circshift(v_DATA,[0,1]);
    end
    DATA = cat(3,DATA,v_DATA);
    
    v_DATA_wave = conj(flip(flip(DATA_wave,1),2)); 
    if mod(size(v_DATA_wave,1),2) == 0
        v_DATA_wave = circshift(v_DATA_wave,[1,0]);
    end
    if mod(size(v_DATA_wave,2),2) == 0
        v_DATA_wave = circshift(v_DATA_wave,[0,1]);
    end
    DATA_wave = cat(3,DATA_wave,v_DATA_wave);
    
    v_samp_mat = conj(flip(flip(samp_mat,1),2)); 
    if mod(size(v_samp_mat,1),2) == 0
        v_samp_mat = circshift(v_samp_mat,[1,0]);
    end
    if mod(size(v_samp_mat,2),2) == 0
        v_samp_mat = circshift(v_samp_mat,[0,1]);
    end
    samp_mat = cat(3,samp_mat,v_samp_mat);
    
    v_samp_mat_wave = conj(flip(flip(samp_mat_wave,1),2)); 
    if mod(size(v_samp_mat_wave,1),2) == 0
        v_samp_mat_wave = circshift(v_samp_mat_wave,[1,0]);
    end
    if mod(size(v_samp_mat_wave,2),2) == 0
        v_samp_mat_wave = circshift(v_samp_mat_wave,[0,1]);
    end
    samp_mat_wave = cat(3,samp_mat_wave,v_samp_mat_wave);

    PsfY = cat(3,PsfY,conj(circshift(flip(PsfY,1),[1,0]))); % please note
    Nc = Nc * 2; % coils doubled
end


% =========================================================================
% Sensitivity Esimatioon
% =========================================================================
kSize = [8,8];
%kSize = [6,6];

if useVCC == 0
    eigThresh_1 = 0.025;
    eigThresh_2 = 0.85;
else
    eigThresh_1 = 0.015;
    eigThresh_2 = 0.75;
end

tic;
[K,S] = dat2Kernel(kCalib, kSize);
idx = find(S >= S(1)*eigThresh_1, 1, 'last');
[M,W] = kernelEig(K(:,:,:,1:idx), [Nx,Ny]);
toc;

%Nnum = 1;
% =========================================================================
% Reconstruction
% =========================================================================
if useVCC == 0 % SENSE recon
    if num_maps == 1
         maps = M(:,:,:,end) .* repmat(W(:,:,end) > eigThresh_2, [1,1,Nc]);
         for n = 1:Nnum
            img_it_wave(:,:,n) = ismrm_cartesian_iterative_Wave(squeeze(DATA_wave(:,:,:,n)), samp_mat, maps, PsfY);
         end
         %as(img_it_wave_1map)
    elseif num_maps == 2
        maps = M(:,:,:,end-num_maps+1:end);

        weights = W(:,:,end-num_maps+1:end);
        weights = (weights - eigThresh_2)./(1-eigThresh_2).* (W(:,:,end-num_maps+1:end) > eigThresh_2);
        weights = -cos(pi*weights)/2 + 1/2;
        nIterCG = 100;
        %nIterCG = 50;
    
        ESP = ESPIRiT(maps, weights);
        WaveESP = WaveESPIRiT(maps, weights, [], PsfY);
        for n = 1:Nnum
            [reskWaveESPIRiT, resWaveESPIRiT] = cgESPIRiT(squeeze(DATA_wave(:,:,:,n)), WaveESP, nIterCG, 0.0, zeros(Nx,Ny,Nc));
            img_it_wave(:,:,n) = sum(resWaveESPIRiT,3);
        end
        %as(img_it_wave_2map);
    end
elseif useVCC == 1
     if num_maps == 2
        maps = M(:,:,:,end-num_maps+1:end);

        weights = W(:,:,end-num_maps+1:end);
        weights = (weights - eigThresh_2)./(1-eigThresh_2).* (W(:,:,end-num_maps+1:end) > eigThresh_2);
        weights = -cos(pi*weights)/2 + 1/2;
        nIterCG = 100;
    
        ESP = ESPIRiT(maps, weights);
        WaveESP = WaveESPIRiT(maps, weights, [], PsfY);
    
        % [reskVCCESPIRiT, resVCCESPIRiT] = cgESPIRiT(DATA, ESP, nIterCG, 0.0, zeros(size(DATA)));
        for n = 1:Nnum
            [reskVCCWaveESPIRiT, resVCCWaveESPIRiT] = cgESPIRiT(squeeze(DATA_wave(:,:,:,n)), WaveESP, nIterCG, 0.0, zeros(Nx,Ny,Nc));
            img_it_vccwave(:,:,n) = sum(resVCCWaveESPIRiT,3);
        end
        %as(img_it_vccwave);
    end
end
%save([file_Wave, sprintf('_res_mapsnum%d_acs%d,.mat', num_maps, acs_len)], 'img_it_wave', 'maps', 'uih_prot');


as(img_it_wave)
%%%%% Registeration %%%%%----------------------------------------------------------------
[RO,PE,PHS] = size(img_it_wave);
Reg = zeros(RO,PE,PHS);
moving = abs(img_it_wave);
fixed = abs(img_it_wave(:,:,1));
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




