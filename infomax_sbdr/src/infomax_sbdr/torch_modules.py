import torch
from torch import nn
from typing import Sequence, Callable

class DinoCustom(nn.Module):
    """ Model with pretrained Dinov2 + dense head
    
    Notes:
        Input to dino should be a (3,224,224) image
    """

    def __init__(
        self,
        dino_hubconf_dir: str,
        dino_pretrained_path: str,
        dense_sizes: Sequence[int],
        output_positions: int,
        input_low_width: int,
        input_low_height: int,
        input_top_width: int,
        input_top_height: int,
        activation:Callable = nn.LeakyReLU(),
        dropout_rate: float = None,
    ):

        super().__init__()

        self.input_low_width = input_low_width
        self.input_low_height = input_low_height
        self.input_top_width = input_top_width
        self.input_top_height = input_top_height
        self.output_positions = output_positions
        self.activation = activation

        self.dino_name = "dinov2_vits14"
        self.dino_out_features = 384
        
        self.dropout_rate = dropout_rate

        # Import the Dino model
        # self.dino = torch.load(dino_pretrained_path, weights_only=True)

        self.dino = torch.hub.load(
            dino_hubconf_dir,
            self.dino_name,
            source="local",
            pretrained=False,
        )

        self.dino.load_state_dict(torch.load(dino_pretrained_path))

        # freeze the parameters of dinov2
        for param in self.dino.parameters():
            param.requires_grad = False

        # initialize the dense layers
        self._initialize_dense_layer(dense_sizes, output_positions)

    
    def _initialize_dense_layer(self, dense_sizes: Sequence[int], output_positions: int):
        self.fc = nn.Sequential()
        in_size = self.dino_out_features * 2
        for i in range(len(dense_sizes) - 1):
            out_size = dense_sizes[i]
            # add a dense layer + activation
            self.fc.add_module(
                f"fc_{i}",
                nn.Sequential(
                    nn.Linear(in_size, out_size),
                    self.activation,
                )
            )
            # dropout, if rate > 0
            if self.dropout_rate > 0:
                self.fc.add_module(
                    f"dropout_{i}",
                    nn.Dropout(self.dropout_rate)
                )

            in_size = out_size

        self.fc.add_module(
            f"fc_{len(dense_sizes)}",
            nn.Linear(in_size, output_positions * 3)
        )

    

    # def _forward_conv_layers(self, x_low: torch.Tensor, x_top: torch.Tensor) -> torch.Tensor:
        
    #     x_low = self.regnet_low(x_low)
    #     x_low = self.flatten(x_low)

    #     x_top = self.regnet_top(x_top)
    #     x_top = self.flatten(x_top)

    #     return torch.cat([x_low, x_top], dim=-1)

    def forward(self, x_low, x_top):
        # pass through dino, both low and top cam images
        x_low = self.dino(x_low)
        x_top = self.dino(x_top)

        # concatenate the features
        x = torch.cat([x_low, x_top], dim=-1)

        # pass through the dense layers
        x = self.fc(x)

        # reshape the output to be in (batch, n_positions, 3) format
        return x.view(-1, self.output_positions, 3)
