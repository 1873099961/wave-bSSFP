% remove oversampling in RO direction
% dim : the dimension of RO in data
function res = rm_ROos(data, dim)
if nargin < 2
    dim = 1;
end
data_1DFT_RO = fftshift(ifft(ifftshift(data,dim),[],dim),dim);
data_size_removeROOS = size(data);
data_size_removeROOS(dim) = data_size_removeROOS(dim)/2;
data_1DFT_RO_center = crop(data_1DFT_RO,data_size_removeROOS);
res = fftshift(fft(ifftshift(data_1DFT_RO_center,dim),[],dim),dim);
