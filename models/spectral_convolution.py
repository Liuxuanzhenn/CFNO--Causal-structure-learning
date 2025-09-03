import torch
import torch.nn as nn

class SubConv(nn.Module):
    """
    A sub-convolutional layer that shares parameters with a main convolutional layer.
    This class is used to extract a specific layer from a multi-layer convolution.
    """
    
    def __init__(self, main_conv, indices):
        """
        Initialize a SubConv layer.
        
        Parameters
        ----------
        main_conv : nn.Module
            The main convolutional layer that contains the parameters.
        indices : int
            The index of the sub-convolutional layer to extract.
        """
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices
        
    def forward(self, x, output_shape=None):
        """
        Forward pass of the sub-convolutional layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        output_shape : tuple, optional
            Output shape for the convolution.
            
        Returns
        -------
        torch.Tensor
            Output of the sub-convolutional layer.
        """
        return self.main_conv(x, indices=self.indices, output_shape=output_shape) 