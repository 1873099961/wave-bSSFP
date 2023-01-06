clear;
% Add path
addpath('utils');
addpath('ESPIRiT_code');
addpath('coilCompression_code');
addpath('ismrm');
addpath('ReadBayData_wave')
addpath('other code')
addpath('nufft_files')
pathname = 'D:\MRI(only modify codes)\UIH_rawdata\2.bssfp gfactor\23.20220428_fengheyun\';
file_Wave = [pathname, 'UID_42092330642979_bwave_locR4dia_amp14_4ch.raw'];

[DATA_wave,~,~,~,noise_scan,maxCh,Data_ACS,Data_Py,Data_WavePy, cTimes] = Read_UIH_Raw_wave(file_Wave);
% [q,w,e,r] = size(DATA_wave); %WC sli=1 is different,reshape as 5D
% DATA_wave = reshape(DATA_wave,q,1,w,e,r); %WC sli=1 is different,reshape as 5D
[q,w,e] = size(DATA_wave); %WC sli=1 phs =1 is different,reshape as 5D
DATA_wave = reshape(DATA_wave,q,1,1,w,e); %WC sli=1 phs =1 is different,reshape as 5D

[q,w,e] = size(Data_ACS); %WC sli=1 phs=1 is different,reshape as 4D
Data_ACS = reshape(Data_ACS,q,1,w,e); %WC sli=1 phs=1 is different,reshape as 4D
[q,w,e] = size(Data_Py); %WC sli=1 phs=1 is different,reshape as 4D
Data_Py = reshape(Data_Py,q,1,w,e); %WC sli=1 phs=1 is different,reshape as 4D
[q,w,e] = size(Data_WavePy); %WC sli=1 phs=1 is different,reshape as 4D
Data_WavePy = reshape(Data_WavePy,q,1,w,e); %WC sli=1 phs=1 is different,reshape as 4D
%% -------------------------------------------------------------------------%
cali_idx = 1;
slice_idx = 1;
Data_Py = permute(squeeze(Data_Py(:,cali_idx,:,:)),[2,1,3]);
Data_WavePy = permute(squeeze(Data_WavePy(:,cali_idx,:,:)),[2,1,3]);
Data_ACS = permute(squeeze(Data_ACS(:,slice_idx,:,:)),[2,1,3]);
% DATA_wave = permute(squeeze(DATA_wave(:,slice_idx,:,:,:)),[3,1,4,2]); % nFE*nPE*ncoil*nNum
DATA_wave = permute(squeeze(DATA_wave(:,slice_idx,:,:,:)),[2,1,3,4]); %WC  sli=1; phs=1;

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

% Ry = 4;
% mask = zeros(Nx,Ny);
% mask(:,floor(Ny/2)+1:-Ry:1) = 1; 
% mask(:,floor(Ny/2)+1:Ry:end) = 1;
% DATA_wave = bsxfun(@times, DATA_wave, mask);     % mask Rfactor
% mask_wave = mask;                                  % Rfactor use this samp_mat
% samp_mat = repmat(mask_wave,[1,1,Nc]);             % Rfactor use this samp_mat
%                            % use Rfactor data samp_mat is only maskRfactor
% samp_mat_wave = repmat(mask_wave,[1,1,Nc]);

% =========================================================================
% Sensitivity Esimatioon
% =========================================================================
kSize = [8,8];
eigThresh_1 = 0.035;
eigThresh_2 = 0.95;

tic;
[K,S] = dat2Kernel(kCalib, kSize);
idx = find(S >= S(1)*eigThresh_1, 1, 'last');
[M,W] = kernelEig(K(:,:,:,1:idx), [Nx,Ny]);
toc;

% =========================================================================
% Reconstruction
% =========================================================================
    if num_maps == 1
         maps = M(:,:,:,end) .* repmat(W(:,:,end) > eigThresh_2, [1,1,Nc]);
         for n = 1:Nnum
            img_it_wave(:,:,n) = ismrm_cartesian_iterative_Wave(squeeze(DATA_wave(:,:,:,n)), samp_mat, maps, PsfY);
         end
    elseif num_maps == 2
        maps = M(:,:,:,end-num_maps+1:end);
        weights = W(:,:,end-num_maps+1:end);
        weights = (weights - eigThresh_2)./(1-eigThresh_2).* (W(:,:,end-num_maps+1:end) > eigThresh_2);
        weights = -cos(pi*weights)/2 + 1/2;
        nIterCG = 50;
        ESP = ESPIRiT(maps, weights);
        WaveESP = WaveESPIRiT(maps, weights, [], PsfY);
        for n = 1:Nnum
            % n = 1;
            [reskWaveESPIRiT, resWaveESPIRiT] = cgESPIRiT(squeeze(DATA_wave(:,:,:,n)), WaveESP, nIterCG, 0.0, zeros(Nx,Ny,Nc));
            img_it_wave(:,:,n) = sum(resWaveESPIRiT,3);
        end
    end
img_it_wave_copy = img_it_wave/max(max(max(abs(ifft2c(img_it_wave)))));
as(img_it_wave_copy)

