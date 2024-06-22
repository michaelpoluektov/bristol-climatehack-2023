from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        # He initialization
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif (
        isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv1d)
    ):
        # Example for Conv2d, adjust if you have ConvNet definition elsewhere
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
