# Import required packages
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    # Initialize the constructor
    def __init__(self, tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        # Grab the image, label, and its bounding box coordinates
        image = self.tensors[0][index]
        label = self.tensors[1][index]

        # Transpose the image such that its channel dimension becomes the leading one
        image = image.permute(2, 0, 1)

        # Check to see if we have any image transformations to apply and if so, apply them
        if self.transforms:
            image = self.transforms(image)

        # Return a tuple of the images
        return image, label

    def __len__(self):
        # Return the size of the dataset
        return self.tensors[0].size(0)
