from .panoptic import PanopticCriterion3D, PanopticMatcher3D


def build_criterion(style='panoptic'):
    assert style in {'panoptic'}

    matcher = PanopticMatcher3D()
    criterion = PanopticCriterion3D(matcher)
    return criterion
