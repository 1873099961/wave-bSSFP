clear;
% Add path
addpath('utils');
addpath('ESPIRiT_code');
addpath('coilCompression_code');
addpath('ismrm');
addpath('ReadBayData_wave');
addpath('other code');
%% retrosperspective undersampled ================================================
pathname = 'D:\MRI(only modify codes)\UIH_rawdata\2.bssfp gfactor\23.20220428_fengheyun\';
file_Wave = [pathname, 'UID_42092330642565_bwave_T2CS4dia_amp0.raw'];

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
%%%%%%%% Reconstruction Parameters %%%%%%%%%%%%%%%%%%%
[RO,PE,~,phs] = size(DATA_wave);
recon = zeros(RO,PE,phs);
for i = 1:phs
    % i = 1;
    data = DATA_wave(:,:,:,i);
    [RO ,PE ,CHA] = size(data);
    % mask = ones(RO,PE,CHA);  % if kspace is not full,use ones mask
    mask = data~=0;
    
    kSize = [8,8];
    eigThresh_1 = 0.025;
    eigThresh_2 = 0.88;
    [K,S] = dat2Kernel(kCalib, kSize);
    idx = find(S >= S(1)*eigThresh_1, 1, 'last');
    [M,W] = kernelEig(K(:,:,:,1:idx), [Nx,Ny]);
    maps1 = M(:,:,:,end) .* repmat(W(:,:,end) > eigThresh_2, [1,1,Nc]);
    maps2 = M(:,:,:,end-1:end);
    weights = W(:,:,end-1:end);
    weights = (weights - eigThresh_2)./(1-eigThresh_2).* (W(:,:,end-1:end) > eigThresh_2);
    weights = -cos(pi*weights)/2 + 1/2;
    %% L1 norm
    % addpath(strcat(pwd,'/utils'));
    % if exist('FWT2_PO') <2
    % 	error('must have Wavelab installed and in the path');
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % L1 Recon Parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N = size(data); 	% image Size
    DN = size(data); 	% data Size
    TVWeight = 0.001; 	% Weight for TV penalty0.002
    xfmWeight = 0.0001;	% Weight for Transform L1 penalty0.005
    Itnlim = 8;		% Number of iterations8
    %generate Fourier sampling operator
    maps = maps1;
    FT = WAVEp2DFT(mask, N, 1, 2, maps, PsfY);     %1maps %mask and N have same dim.
%     maps = maps2;
%     FT = WAVEp2DFT(mask, N, 1, 2, maps, PsfY, weights);  %2maps %mask and N have same dim.
    % scale data
    %im_dc = FT'*(data.*mask);
    im_dc = FT'*data;
    data = data/max(abs(im_dc(:)));
    im_dc = im_dc/max(abs(im_dc(:)));
    %generate transform operator
    % XFM = Wavelet('Daubechies',4,4);	% Wavelet
    % XFM = ESPIRiT(maps,data(:,:,1)~=0);
    XFM = 1;
    % initialize Parameters for reconstruction
    param = init;
    param.FT = FT;
    param.XFM = XFM;
    param.TV = TVOP;
    param.data = data;
    param.TVWeight =TVWeight;     % TV penalty
    param.xfmWeight = xfmWeight;  % L1 wavelet penalty
    param.Itnlim = Itnlim;
    
    [sx,sy,nc,nv] = size(maps);
    res = XFM*im_dc;
    % do iterations
    tic
    for n=1:5
        res = fnlCg(res,param);
        im_res = XFM'*res;
    end
    toc
    recon1(:,:,i) = im_res(:,:,nv);
end
recon1 = recon1/max(max(max(abs(ifft2c(recon1)))));
as(recon1)