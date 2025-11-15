from yacs.config import CfgNode


def add_custom_config(_C: CfgNode):
    _C.MODEL.USE_MATCHNESS = False
    _C.MODEL.NUM_ITER = 2
    _C.MODEL.WITH_AFFINITY = False
    _C.MODEL.AFFINITY_CONVS = 4
    _C.MODEL.AFFINITY_FOR_MASK = False
    _C.MODEL.INIT_MASK_METHOD = 'connected_components'
    _C.MODEL.NUM_MASKS = 100
    _C.MODEL.NUM_LEARNED_MASKS = 100
    _C.MODEL.RETURN_INIT_MASK = False
    _C.MODEL.AUX_INST_DECODER = False
    _C.MODEL.WITH_AGFP = False
    _C.MODEL.INFERENCE_WITHOUT_BG = False
    _C.MODEL.INST_DECODER_SHARE_WEIGHTS = False
    _C.MODEL.INST_DECODER_BG_CONV_AS_BIAS = False   # use only a bias as bgconv
    _C.MODEL.INST_DECODER_BG_CONV_SHARE = False
    _C.MODEL.INST_DECODER_BG_CONV_NORM = 'bn'
    _C.MODEL.FEED_GT_MASK = False
