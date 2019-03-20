"""Faster R-CNN and Mask R-CNN operations."""
from . import _make


def roi_align(data, rois, pooled_size, spatial_scale, sample_ratio=-1, layout='NCHW'):
    """ROI align operator.

    Parameters
    ----------
    data : relay.Expr
        4-D tensor with shape [batch, channel, height, width]

    rois : relay.Expr
        2-D tensor with shape [num_roi, 5]. The last dimension should be in format of
        [batch_index, w_start, h_start, w_end, h_end]

    pooled_size : list/tuple of two ints
        output size

    spatial_scale : float
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        of total stride in convolutional layers, which should be in range (0.0, 1.0]

    sample_ratio : int
        Optional sampling ratio of ROI align, using adaptive size by default.

    Returns
    -------
    output : relay.Expr
        4-D tensor with shape [num_roi, channel, pooled_size, pooled_size]
    """
    return _make.roi_align(data, rois, pooled_size, spatial_scale, sample_ratio, layout)


def roi_pool(data, rois, pooled_size, spatial_scale, layout='NCHW'):
    """ROI pool operator.

    Parameters
    ----------
    data : relay.Expr
        4-D tensor with shape [batch, channel, height, width]

    rois : relay.Expr
        2-D tensor with shape [num_roi, 5]. The last dimension should be in format of
        [batch_index, w_start, h_start, w_end, h_end]

    pooled_size : list/tuple of two ints
        output size

    spatial_scale : float
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        of total stride in convolutional layers, which should be in range (0.0, 1.0]

    Returns
    -------
    output : relay.Expr
        4-D tensor with shape [num_roi, channel, pooled_size, pooled_size]
    """
    return _make.roi_pool(data, rois, pooled_size, spatial_scale, layout)


def proposal(cls_prob,
             bbox_pred,
             im_info,
             scales,
             ratios,
             feature_stride,
             threshold,
             rpn_pre_nms_top_n,
             rpn_post_nms_top_n,
             rpn_min_size,
             iou_loss):
    """Proposal operator.

    Parameters
    ----------
    cls_prob : relay.Expr
        4-D tensor with shape [batch, 2 * num_anchors, height, width].

    bbox_pred : relay.Expr
        4-D tensor with shape [batch, 4 * num_anchors, height, width].

    im_info : relay.Expr
        2-D tensor with shape [batch, 3]. The last dimension should be in format of
        [im_height, im_width, im_scale]

    scales : list/tuple of float
        Scales of anchor windoes.

    ratios : list/tuple of float
        Ratios of anchor windoes.

    feature_stride : int
        The size of the receptive field each unit in the convolution layer of the rpn, for example
        the product of all stride's prior to this layer.

    threshold : float
        Non-maximum suppression threshold.

    rpn_pre_nms_top_n : int
        Number of top scoring boxes to apply NMS. -1 to use all boxes.

    rpn_post_nms_top_n : int
        Number of top scoring boxes to keep after applying NMS to RPN proposals.

    rpn_min_size : int
        Minimum height or width in proposal.

    iou_loss : bool
        Usage of IoU loss.

    Returns
    -------
    output : relay.Expr
        2-D tensor with shape [batch * rpn_post_nms_top_n, 5]. The last dimension is in format of
        [batch_index, w_start, h_start, w_end, h_end].
    """
    return _make.proposal(cls_prob, bbox_pred, im_info, scales, ratios, feature_stride, threshold,
                          rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_min_size, iou_loss)
