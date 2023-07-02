% zero padding the rawdata 
function data = raw_zeroPading_end(data, dim, N)
 N_cut = size(data, dim);
 if N_cut ~= N
    padsize = zeros(1, ndims(data));
    padsize(dim) = N - N_cut;
    data = padarray(data, padsize, 0, 'post');
 end
 