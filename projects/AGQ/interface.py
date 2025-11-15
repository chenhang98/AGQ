
class E2EMixin:
    """To parse the config by module rather than by `build_model`"""

    @staticmethod
    def parse_config(cfg, kwargs):
        return kwargs


class AutoLossMixin:
    """Used to calculate target and loss by module itself, instead of by loss module."""

    def get_loss(self, label):
        return {}

    def get_target(self, label):
        return None
