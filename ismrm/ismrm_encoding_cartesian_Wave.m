function outp =  ismrm_encoding_cartesian_Wave(inp,csm,sampling_mask,PsfY,transpose_indicator)

% scale = numel(sampling_mask)/sum(sampling_mask(:) > 0);
scale = 1; % do not scale QZL 2021/9/9

if (strcmp(transpose_indicator,'transp'))
    outp = zeros(size(sampling_mask));
    % outp(repmat(sampling_mask,[1 1 size(csm,3)]) == 1) = inp(:);
    outp(sampling_mask == 1) = inp(:); % QZL VCC
    % QZL wave
    outp = ifftc(outp, 2);
    outp = outp .* conj(PsfY);
    outp = fftc(outp, 2);
    % END
    outp = ismrm_transform_kspace_to_image(outp,[1,2])*sqrt(scale);
    outp = crop(outp,size(csm)); % crop
    outp = sum(conj(csm) .* outp,3);
    outp = outp(:);
elseif (strcmp(transpose_indicator, 'notransp'))
    outp = repmat(reshape(inp,size(csm,1),size(csm,2)),[1 1 size(csm,3)]) .* csm;
    outp = zpad(outp, size(PsfY)); % zpad
    % QZL wave
    outp = fftc(outp, 1);
    outp = outp .* PsfY;
    outp = ifftc(outp, 1);
    % END
    outp = ismrm_transform_image_to_kspace(outp, [1,2])*sqrt(scale);
    % outp = outp(repmat(sampling_mask,[1 1 size(csm,3)]) == 1);
    outp = outp(sampling_mask == 1); % QZL VCC
else
    error('Transpose flag not appropriately defined');
end

return
