function [img,snr,g,noise_psf] = ismrm_cartesian_iterative_SENSE_QZL(data,samp_mat,csm,reg,replicas)

if nargin < 4
    reg = [];
end

if nargin < 5
    replicas = 100;
end

inp = data(samp_mat == 1);
csm_sq = sum(csm .* conj(csm),3); 
csm_sq(csm_sq < eps) = 1;

M = spdiag(sqrt(csm_sq)); %Preconditioner

if (isempty(reg))
    E = @(x,tr) ismrm_encoding_cartesian_SENSE_QZL(x,csm,samp_mat,tr);
    reg_out = [];
else
    % QZL regularization
    reg(reg == 0) = 1; 
    reg_mask = 1./reg;
    reg_mask = reg_mask * (numel(reg_mask)/sum(reg_mask(:)));
    reg_mask = reg_mask * 0.5;
    % END
    
    E_base = @(x,tr) ismrm_encoding_cartesian_SENSE_QZL(x,csm,samp_mat,tr);
    E = @(x,tr) regularized_E(x,E_base,reg_mask,tr);
    reg_out = zeros(numel(reg_mask),1);
    M = M + spdiag(reg_mask(:));
end

nIterCG = 100; 
% [img,FLAG,RELRES,ITER,RESVEC] = lsqr(E, [inp(:); reg_out], 1e-3, nIterCG, M);
[img,FLAG,RELRES,ITER,RESVEC] = lsqr(E, [inp(:); reg_out], 1e-3, nIterCG);
img = reshape(img,size(csm,1),size(csm,2));

if (nargout > 1)
    image_formation_func = @(x) reshape(lsqr(E,[x; reg_out],1e-3,nIterCG),[size(csm,1) size(csm,2)]);
    [snr,g,noise_psf] = ismrm_pseudo_replica(inp(:),image_formation_func,replicas);
    csm_sq = sum(csm .* conj(csm),3); csm_sq(csm_sq < eps) = 1;
    g = g .* sqrt(csm_sq);
end

return

function out = regularized_E(x,E,reg_mask,transpose_indicator)
    numimgel = length(reg_mask(:));
    if (strcmp(transpose_indicator,'transp'))
        numkel = length(x(:))-numimgel;
        out = E(x(1:numkel),transpose_indicator) + reg_mask(:).*x((numkel+1):end);
    elseif (strcmp(transpose_indicator, 'notransp'))
        out = [E(x,transpose_indicator);reg_mask(:).*x];
    else
        error('Transpose flag not appropriately defined');
    end
return



