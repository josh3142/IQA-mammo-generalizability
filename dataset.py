from torchvision import transforms
from torch.utils.data import Dataset

from typing import Callable, Optional, List, Tuple, Union

from mammo import Mammo_h5
import cv2


def get_dataset(name: str, path: str, types: Optional[List[str]], 
    transform: Optional[Callable], 
    margins: Optional[List[int]] = None) -> Dataset:
    """
    The datasets load h5 datafiles and extract the data from it. 
    """
    if name in ["area", "cubic", "nearest", "linear", "lanczos"]:
        data = Mammo_h5(path, transform, margins)
    else:
        raise NotImplementedError
        
    return data

def resize_data(method: str, size_x: int, size_y: int) -> Callable:
    """
    Resize the input images to a given size with a specific resizing method.
    """
    if method == "area":  
        resize = lambda img: cv2.resize(img, dsize = (size_x, size_y),
            interpolation = cv2.INTER_AREA)
    elif method == "linear":  
        resize = lambda img: cv2.resize(img, dsize = (size_x, size_y),
            interpolation = cv2.INTER_LINEAR)
    elif method == "nearest":  
        resize = lambda img: cv2.resize(img, dsize = (size_x, size_y),
            interpolation = cv2.INTER_NEAREST)
    elif method == "cubic":  
        resize = lambda img: cv2.resize(img, dsize = (size_x, size_y),
            interpolation = cv2.INTER_CUBIC)
    elif method == "lanczos":  
        resize = lambda img: cv2.resize(img, dsize = (size_x, size_y),
            interpolation = cv2.INTER_LANCZOS4)
    else:
        raise NotImplementedError("This resize method is not implemented.")

    return resize

def data_transformation_mammo(name: str, method: str, 
    size: Union[List, Tuple], are_imgs_augmented: bool = False) -> Callable:
    " Data transformation rule of the non-resized images "
    # normalization from DOI 10.1088/2632-2153/ac7a03
    normalize = transforms.Normalize(mean = (2520.8828,),
                                     std = (3022.4248, ))
    
    if "cnn" in name:
        if are_imgs_augmented:
            trafo = [
                resize_data(method, size_x = size[0], size_y = size[1]),
                transforms.ToTensor(),
                transforms.RandomVerticalFlip(p = 0.5),
                transforms.RandomHorizontalFlip(p = 0.5),
                normalize
            ]
        else:
            trafo = [ 
                resize_data(method, size_x = size[0], size_y = size[1]),
                transforms.ToTensor(),
                normalize
                ]
    elif name in ["resnet18", "densenet121", "vgg13bn", "efficientnets"]: 
        if are_imgs_augmented:
            trafo = [
                resize_data(method, size_x = size[0], size_y = size[1]),
        # Grayscale needs PIL image or Tensor with correct number of channels. 
                transforms.ToTensor(),
                normalize,
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels = 3),
                transforms.ToTensor(),
            ]
        else:
            trafo = [
                resize_data(method, size_x = size[0], size_y = size[1]),
                transforms.ToTensor(),
                normalize,
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels = 3),
                transforms.ToTensor()
            ]
    return transforms.Compose(trafo)


def data_transformation(data_name: str, name: str, size: Union[List, Tuple],
    are_imgs_augmented: bool = False) -> Callable:
    "Transformation rule from the resized data that stored in h5 datafiles"
    # normalization from DOI 10.1088/2632-2153/ac7a03
    normalize = transforms.Normalize(mean = (2520.8828,),
                                     std = (3022.4248, ))
 
    if data_name in ["area", "cubic", "nearest", "linear", "lanczos"]:
        if "cnn" in name:
            if are_imgs_augmented:
                trafo = [normalize,
                         transforms.RandomVerticalFlip(p = 0.5),
                         transforms.RandomHorizontalFlip(p = 0.5),
                         transforms.RandomResizedCrop(
                            size  = (size[0], size[1]),
                            scale = (0.9, 1.0))
                         ]
            else:
                trafo = [normalize]
        elif name in ["resnet18", "densenet121", "vgg13bn", "efficientnets"]: 
            if are_imgs_augmented:
                trafo = [normalize, 
                        transforms.ToPILImage(),
                        transforms.Grayscale(num_output_channels = 3),
                        transforms.ToTensor(),
                        transforms.RandomVerticalFlip(p = 0.5),
                        transforms.RandomHorizontalFlip(p = 0.5),
                        transforms.RandomResizedCrop(
                            size  = (size[0], size[1]),
                            scale = (0.9, 1.0))
                    ]
            else:
                trafo = [normalize, 
                        transforms.ToPILImage(),
                        transforms.Grayscale(num_output_channels = 3),
                        transforms.ToTensor()
                        ]
    else:
        raise NotImplementedError
    
    return transforms.Compose(trafo)