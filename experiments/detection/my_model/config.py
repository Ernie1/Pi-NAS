def add_my_config(cfg):
    cfg.MODEL.BACKBONE.MODEL = None
    cfg.MODEL.BACKBONE.PRETRAINED = None
    # Apply deep stem 
    cfg.MODEL.RESNETS.DEEP_STEM = False
    # Apply avg after conv2 in the BottleBlock
    # When AVD=True, the STRIDE_IN_1X1 should be False
    cfg.MODEL.RESNETS.AVD = False
    # Apply avg_down to the downsampling layer for residual path 
    cfg.MODEL.RESNETS.AVG_DOWN = False
    # choice_indices in ResNeSt
    cfg.MODEL.RESNETS.CHOICE_INDICES = None