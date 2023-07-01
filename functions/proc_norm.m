function res = proc_norm(im, mask_roi)
im = abs(im);
im = bsxfun(@times, mask_roi, im);
Nnum = size(im, 3);
res = []; idx = 1;
for n = 1:Nnum
    if( sum(sum(im(:,:,n))) ~=0)
        %IMG = im(:,:,n);   N3_v2;  res(:,:,idx) = img;
        res(:,:,idx) = im(:,:,n);
        idx = idx + 1;
    end
end
mask_roi = repmat(mask_roi, [1,1,size(res, 3)]);
res = res./ mean(res(mask_roi == 1));
