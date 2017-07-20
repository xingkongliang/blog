---
title: Faster R-CNN代码解析
layout: post
tags: [Deep Learning]
---

有关caffe的函数：
```
iter_ = caffe_solver.iter();
max_iter = caffe_solver.max_iter();
caffe_solver.net.set_phase('train');

caffe_solver.net.reshape_as_input(net_inputs);

% one iter SGD update
caffe_solver.net.set_input_data(net_inputs);
caffe_solver.step(1);

rst = caffe_solver.net.get_output();


```

### fast_rcnn_prepare_image_roidb

    [image_roidb_train, bbox_means, bbox_stds]...
                            = fast_rcnn_prepare_image_roidb(conf, opts.imdb_train, opts.roidb_train);


### append_bbox_regression_targets

主要计算image_roidb(i).bbox_targets，means和stds。

```
function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb, means, stds)
    % means and stds -- (k+1) * 4, include background class

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    valid_imgs = true(num_images, 1);
    for i = 1:num_images
       rois = image_roidb(i).boxes; 
       % 
       [image_roidb(i).bbox_targets, valid_imgs(i)] = ...
           compute_targets(conf, rois, image_roidb(i).overlap);
    end
    if ~all(valid_imgs)
        image_roidb = image_roidb(valid_imgs);
        num_images = length(image_roidb);
        fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
    end
        
    if ~(exist('means', 'var') && ~isempty(means) && exist('stds', 'var') && ~isempty(stds))
        % Compute values needed for means and stds
        % var(x) = E(x^2) - E(x)^2
        class_counts = zeros(num_classes, 1) + eps;
        sums = zeros(num_classes, 4);
        squared_sums = zeros(num_classes, 4);
        for i = 1:num_images
           targets = image_roidb(i).bbox_targets;
           for cls = 1:num_classes
              cls_inds = find(targets(:, 1) == cls);
              if ~isempty(cls_inds)
                 class_counts(cls) = class_counts(cls) + length(cls_inds); 
                 sums(cls, :) = sums(cls, :) + sum(targets(cls_inds, 2:end), 1);
                 squared_sums(cls, :) = squared_sums(cls, :) + sum(targets(cls_inds, 2:end).^2, 1);
              end
           end
        end

        means = bsxfun(@rdivide, sums, class_counts);
        stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;
        
        % add background class
        means = [0, 0, 0, 0; means]; 
        stds = [0, 0, 0, 0; stds];
    end
    
    % Normalize targets
    for i = 1:num_images
        targets = image_roidb(i).bbox_targets;
        for cls = 1:num_classes
            cls_inds = find(targets(:, 1) == cls);
            if ~isempty(cls_inds)
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@minus, image_roidb(i).bbox_targets(cls_inds, 2:end), means(cls+1, :));
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@rdivide, image_roidb(i).bbox_targets(cls_inds, 2:end), stds(cls+1, :));
            end
        end
    end
end
```

### compute_targets
```
function [bbox_targets, is_valid] = compute_targets(conf, rois, overlap)

    overlap = full(overlap);

    [max_overlaps, max_labels] = max(overlap, [], 2);

    % ensure ROIs are floats
    rois = single(rois);
    
    bbox_targets = zeros(size(rois, 1), 5, 'single');
    
    % Indices of ground-truth ROIs
    gt_inds = find(max_overlaps == 1);
    
    if ~isempty(gt_inds)
        % Indices of examples for which we try to make predictions
        ex_inds = find(max_overlaps >= conf.bbox_thresh);

        % Get IoU overlap between each ex ROI and gt ROI
        ex_gt_overlaps = boxoverlap(rois(ex_inds, :), rois(gt_inds, :));

        assert(all(abs(max(ex_gt_overlaps, [], 2) - max_overlaps(ex_inds)) < 10^-4));

        % Find which gt ROI each ex ROI has max overlap with:
        % this will be the ex ROI's gt target
        [~, gt_assignment] = max(ex_gt_overlaps, [], 2);
        gt_rois = rois(gt_inds(gt_assignment), :);
        ex_rois = rois(ex_inds, :);

        % need to change
        [regression_label] = fast_rcnn_bbox_transform(ex_rois, gt_rois);

        bbox_targets(ex_inds, :) = [max_labels(ex_inds), regression_label];
    end
    
    % Select foreground ROIs as those with >= fg_thresh overlap
    is_fg = max_overlaps >= conf.fg_thresh;
    % Select background ROIs as those within [bg_thresh_lo, bg_thresh_hi)
    is_bg = max_overlaps < conf.bg_thresh_hi & max_overlaps >= conf.bg_thresh_lo;
    
    % check if there is any fg or bg sample. If no, filter out this image
    is_valid = true;
    if ~any(is_fg | is_bg)
        is_valid = false;
    end
end
```

### generate_random_minibatch

        [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, conf.ims_per_batch);



### fast_rcnn_bbox_transform.m

ex_boxes为与ground truth的overlap大于0.5的窗口，gt_boxes为ex_boxes对应的ground truth。
计算ex_boxes的回归值。

```
function [regression_label] = fast_rcnn_bbox_transform(ex_boxes, gt_boxes)
% [regression_label] = fast_rcnn_bbox_transform(ex_boxes, gt_boxes)
% --------------------------------------------------------

    ex_widths = ex_boxes(:, 3) - ex_boxes(:, 1) + 1;
    ex_heights = ex_boxes(:, 4) - ex_boxes(:, 2) + 1;
    ex_ctr_x = ex_boxes(:, 1) + 0.5 * (ex_widths - 1);
    ex_ctr_y = ex_boxes(:, 2) + 0.5 * (ex_heights - 1);
    
    gt_widths = gt_boxes(:, 3) - gt_boxes(:, 1) + 1;
    gt_heights = gt_boxes(:, 4) - gt_boxes(:, 2) + 1;
    gt_ctr_x = gt_boxes(:, 1) + 0.5 * (gt_widths - 1);
    gt_ctr_y = gt_boxes(:, 2) + 0.5 * (gt_heights - 1);
    
    targets_dx = (gt_ctr_x - ex_ctr_x) ./ (ex_widths+eps);
    targets_dy = (gt_ctr_y - ex_ctr_y) ./ (ex_heights+eps);
    targets_dw = log(gt_widths ./ ex_widths);
    targets_dh = log(gt_heights ./ ex_heights);
    
    regression_label = [targets_dx, targets_dy, targets_dw, targets_dh];
end
```

之后又计算了bbox_targets的均值和方差，对bbox_targets进行了归一化。

```
function [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_blob] = fast_rcnn_get_minibatch(conf, image_roidb)
% [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_blob] ...
%    = fast_rcnn_get_minibatch(conf, image_roidb)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    % Sample random scales to use for each image in this batch
    random_scale_inds = randi(length(conf.scales), num_images, 1);
    
    assert(mod(conf.batch_size, num_images) == 0, ...
        sprintf('num_images %d must divide BATCH_SIZE %d', num_images, conf.batch_size));
    
    % 每幅图像取64个rois
    rois_per_image = conf.batch_size / num_images; % conf.batch_size=128, num_images=2
    % 每个图像正样本的rois为16个， 这里conf.fg_fraction=0.25
    fg_rois_per_image = round(rois_per_image * conf.fg_fraction);
    
    % Get the input image blob
    % 图像最小的边要大于600，并且最大的边要小于1000。
    % 返回的im_blob为4D，[w, h, 3, 2]， im_scalse为图像的缩放因子。
    [im_blob, im_scales] = get_image_blob(conf, image_roidb, random_scale_inds);
    
    % build the region of interest and label blobs
    rois_blob = zeros(0, 5, 'single');
    labels_blob = zeros(0, 1, 'single');
    bbox_targets_blob = zeros(0, 4 * (num_classes+1), 'single');
    bbox_loss_blob = zeros(size(bbox_targets_blob), 'single');
    
    for i = 1:num_images
    % 这步比较重要，采样正例样本和反例样本
        [labels, ~, im_rois, bbox_targets, bbox_loss] = ...
            sample_rois(conf, image_roidb(i), fg_rois_per_image, rois_per_image);
        
        % Add to ROIs blob
        % 对所有选取的ROIs做尺度变换，缩放到对应图像缩放的尺度。
        feat_rois = fast_rcnn_map_im_rois_to_feat_rois(conf, im_rois, im_scales(i));
        batch_ind = i * ones(size(feat_rois, 1), 1);
        rois_blob_this_image = [batch_ind, feat_rois];
        rois_blob = [rois_blob; rois_blob_this_image];
        
        % Add to labels, bbox targets, and bbox loss blobs
        labels_blob = [labels_blob; labels];
        bbox_targets_blob = [bbox_targets_blob; bbox_targets];
        bbox_loss_blob = [bbox_loss_blob; bbox_loss];
    end
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3, 4]));
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = single(permute(rois_blob, [3, 4, 2, 1]));
    labels_blob = single(permute(labels_blob, [3, 4, 2, 1]));
    bbox_targets_blob = single(permute(bbox_targets_blob, [3, 4, 2, 1])); 
    bbox_loss_blob = single(permute(bbox_loss_blob, [3, 4, 2, 1]));
    
    assert(~isempty(im_blob));
    assert(~isempty(rois_blob));
    assert(~isempty(labels_blob));
    assert(~isempty(bbox_targets_blob));
    assert(~isempty(bbox_loss_blob));
end
```

### sample_rois



```
%% Generate a random sample of ROIs comprising foreground and background examples.
function [labels, overlaps, rois, bbox_targets, bbox_loss_weights] = ...
    sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image)

    [overlaps, labels] = max(image_roidb(1).overlap, [], 2);
%     labels = image_roidb(1).max_classes;
%     overlaps = image_roidb(1).max_overlaps;
    rois = image_roidb(1).boxes;
    
    % Select foreground ROIs as those with >= FG_THRESH overlap

    % 选取Overlap大于0.5的样本作为正例样本
    fg_inds = find(overlaps >= conf.fg_thresh);
    % Guard against the case when an image has fewer than fg_rois_per_image
    % foreground ROIs
    fg_rois_per_this_image = min(fg_rois_per_image, length(fg_inds));
    % Sample foreground regions without replacement
    if ~isempty(fg_inds)
       fg_inds = fg_inds(randperm(length(fg_inds), fg_rois_per_this_image));
    end
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    % 选择Overlap在0.1~0.5之间的样本作为反例样本
    bg_inds = find(overlaps < conf.bg_thresh_hi & overlaps >= conf.bg_thresh_lo);
    % Compute number of background ROIs to take from this image (guarding
    % against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image;
    bg_rois_per_this_image = min(bg_rois_per_this_image, length(bg_inds));
    % Sample foreground regions without replacement
    if ~isempty(bg_inds)
       bg_inds = bg_inds(randperm(length(bg_inds), bg_rois_per_this_image));
    end
    % The indices that we're selecting (both fg and bg)
    keep_inds = [fg_inds; bg_inds];
    % Select sampled values from various arrays
    labels = labels(keep_inds);
    % Clamp labels for the background ROIs to 0
    labels((fg_rois_per_this_image+1):end) = 0;
    overlaps = overlaps(keep_inds);
    rois = rois(keep_inds, :);
    
    assert(all(labels == image_roidb.bbox_targets(keep_inds, 1)));
    
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    
    [bbox_targets, bbox_loss_weights] = get_bbox_regression_labels(conf, ...
        image_roidb.bbox_targets(keep_inds, :), num_classes);
    
end
```


### loss_bbox

layer {
    name: "loss_bbox"
    type: "SmoothL1Loss"
    bottom: "bbox_pred"
    bottom: "bbox_targets"
    bottom: "bbox_loss_weights"
    top: "loss_bbox"
    loss_weight: 1
}

```
        net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};
        caffe_solver.net.reshape_as_input(net_inputs);
```
