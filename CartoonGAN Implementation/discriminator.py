import torch
import torch.nn as nn

class Block(nn.Module):
    # init = initialize constructor
    # self = current instance of an object of said class
    # stride = number of pixels a filter will skip over during convolution 
    def __init__(self, in_channels, out_channels, stride):
        # call parent class' initialization function using super? 
        super().__init__() 

        # Sequential = container that modules can be added to

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # Conv2d applies 2D convolution over an input (in_channels)

        # InstanceNorm2d applies instance normalization to output
        # Normalization/feature scaling -> used to scale all features to a similar range so that no feature dominates due to having a lower/higher value than the rest of them
        # instance normalization -> computes mean/standard deviation and normalize across each channel in each training

        # LeakyReLu = activation function, used to convert each neuron into something of value, "rectified linear unit", non-linear function, introduces small gradient for negative inputs to reduce dying ReLU problem

        # creates a convolution block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias = True, padding_mode = "reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    # chains/adds new things to previous sequential block
    def forward(self, x):
        return self.conv(x)
        

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, features = [64,128,256,512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),

        )

        # create layers for discriminator
        layers = []

        # initial block has 64 channels
        in_channels = features[0]
        for feature in features[1:]:
            # use stride of 2 for all blocks except the last one
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        
        # output single value for discriminator 
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial(x)
        
        #use sigmoid activation function to make sure final value is between 0 and 1 
        return torch.sigmoid(self.model(x))


# test if discrimiantor works  
# def test():
#     x=torch.randn((5,3,256,256))
#     model=Discriminator(in_channels=3)
#     preds = model(x)
#     print(model)
#     print(preds.shape)

# if __name__  == "__main__":
#     test()