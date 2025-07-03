import torch
import torch.nn as nn
import torch.nn.functional as F
# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

''' Swish activation '''
class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


''' MLP '''
class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out



''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat

class SimplifiedConvNet(nn.Module):
    def __init__(self, channel=1, num_classes=10, net_width=128, net_depth=3, im_size=(32, 32), embedding_size=None):
        super(SimplifiedConvNet, self).__init__()
        
        self.im_size = im_size
        self.features = self._make_layers(channel, net_width, net_depth)
        
        # Calculate output dimensions dynamically by passing a dummy tensor
        with torch.no_grad():
            dummy_input = torch.zeros(1, channel, im_size[0], im_size[1])
            dummy_output = self.features(dummy_input)
            num_feat = dummy_output.view(dummy_output.size(0), -1).size(1)
        
        # Add linear layer to standardize embedding size
        if embedding_size is None:
            embedding_size = num_feat
        self.embedding_layer = nn.Linear(num_feat, embedding_size)
        self.classifier = nn.Linear(embedding_size , num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = self.embedding_layer(out)
        out = self.classifier(out)
        return out
    
    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.embedding_layer(out)
        return out

    def _make_layers(self, channel, net_width, net_depth):
        layers = []
        in_channels = channel
        
        for i in range(net_depth):
            # First layer has special padding (3,3), others have (1,1)
            padding = 3 if channel == 1 and i == 0 else 1
            layers.append(nn.Conv2d(in_channels, net_width, kernel_size=3, stride=1, padding=padding))
            layers.append(nn.GroupNorm(net_width, net_width, affine=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

            in_channels = net_width

        return nn.Sequential(*layers)

# Example usage
if __name__ == '__main__':
    # Test with non-square input
    print("Testing SimplifiedConvNet with non-square input (13, 128):")
    model = SimplifiedConvNet(channel=1, num_classes=10, im_size=(13, 128))
    input_tensor = torch.randn(1, 1, 13, 128)
    output = model(input_tensor)
    embedding = model.embed(input_tensor)
    print("Output shape (SimplifiedConvNet):", output.shape)
    print("Embedding shape (SimplifiedConvNet):", embedding.shape)
    
    # Test with square input
    print("\nTesting SimplifiedConvNet with square input (128, 128):")
    model_square = SimplifiedConvNet(channel=1, num_classes=10, im_size=(128, 128))
    input_tensor_square = torch.randn(1, 1, 128, 128)
    output_square = model_square(input_tensor_square)
    embedding_square = model_square.embed(input_tensor_square)
    print("Output shape (SimplifiedConvNet):", output_square.shape)
    print("Embedding shape (SimplifiedConvNet):", embedding_square.shape)
    
    # Print model architecture
    print("\nModel architecture:")
    print(model)