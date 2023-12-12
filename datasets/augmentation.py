import torchvision.transforms as transforms
import albumentations.pytorch as Ap
import albumentations as A

class BaseAugmentation(object):
    def __init__(self, img_size, is_train):
        self.is_train = is_train
        self.img_size = img_size
        self.transforms = self.get_transforms()
        
    def __call__(self, image, label = None):
        inputs = {"image": image}
        if label is not None:
            inputs["mask"] = label 
        if self.transforms is not None:
            result = self.transforms(**inputs)
            image = result["image"]
            if label is not None:
                label = result["mask"]
        return image, label 
    
    def get_transforms(self):
        if self.is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop((self.img_size,self.img_size)),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size,self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
class CustomAugmentation(BaseAugmentation):
    def get_transforms(self):
        if self.is_train:
            pass 
        else:
            pass

class BaseAlbuAugmentation(object):
    def __init__(self, img_size, is_train):
        self.is_train = is_train
        self.img_size = img_size
        self.transforms = self.get_transforms()
        
    def __call__(self, image):
        inputs = {"image": image}
        
        if self.transforms is not None:
            result = self.transforms(**inputs)             
            image = result["image"]
        return image 
    
    def get_transforms(self):
        if self.is_train:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    Ap.transforms.ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    Ap.transforms.ToTensorV2(),
                ]
            )