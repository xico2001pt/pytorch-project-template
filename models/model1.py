import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, num_classes):
        super(Model1, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(3*32*32, num_classes)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = self.fc(x)
        return x
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_input_size(self):
        return (3, 32, 32)
    
    def get_output_shape(self):
        return (self.num_classes,)
