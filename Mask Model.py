import warnings
warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import pandas
import numpy as np
import os
import cv2
from xml.etree import ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import DataLoader , Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import transforms



    

root_dir = r"D:\deep_learning\Torch\Mask Detection\face-mask-detection"

class DatasetMaker(Dataset):
    def __init__(self , root_dir , transform=None):
        self.image_dir = os.path.join(root_dir, 'images')
        self.annot_dir = os.path.join(root_dir, 'annotations')
        self.transform = transform
        self.image_names = os.listdir(self.image_dir)
        self.image_names.sort()
        
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self , idx):
            image_path = os.path.join(self.image_dir , f"maksssksksss{idx}.png")
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img)
            
            
            target = {}
            annot_path = os.path.join(self.annot_dir, f'maksssksksss{idx}.xml')
            tree = ET.parse(annot_path)
            root = tree.getroot()
    #         size = root.find('size')

            boxes = []
            labels = []
            for obj in root.findall('object'):
                label = obj.find('name').text
                if label == 'with_mask':
                    labels.append(1)
                elif label == 'without_mask':
                    labels.append(2)
                elif label == 'mask_weared_incorrect':
                    labels.append(3)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels)
            target['boxes'] = boxes
            target['labels'] = labels
            return img, target


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485 , 0.456 , 0.406] , std=[0.229, 0.224, 0.225])
])

collate_fn = lambda batch: tuple(zip(*batch))


dataset = DatasetMaker(root_dir=root_dir , transform=transform)
dataloader = DataLoader(dataset ,  shuffle=True, collate_fn=collate_fn , batch_size=32)




# from matplotlib import pyplot as plt
# from matplotlib import patches

# class UnNormalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#         Returns:
#             Tensor: Normalized image.
#         """
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#             # The normalize code -> t.sub_(m).div_(s)
#         return tensor

# unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


# def display_image(image, targets, dataset):
#     fig, ax = plt.subplots(1)
#     # Unnormalize the input image and convert it to a numpy array
#     unnormalized_image = unorm(image).cpu().permute(1, 2, 0).numpy()
#     ax.imshow(unnormalized_image)
#     boxes = targets['boxes']
#     labels = targets['labels']
#     label_color = ['green', 'red', 'purple' ]
#     for j, box in enumerate(boxes):
#         xmin, ymin, xmax, ymax = box
#         rect = patches.Rectangle((xmin, ymin),( xmax - xmin), (ymax - ymin), linewidth=2, edgecolor=label_color[labels[j] - 1], facecolor='none')
#         ax.add_patch(rect)
# #         ax.text(xmin, ymin - 10, label_names[labels[j] - 1], fontsize=8, color='r')
#     plt.show()

# for i, batch in enumerate(dataloader):
#     if i == 3: # display images from the 4th batch
#         for j in range(len(batch[0])):
#             image = batch[0][j]
#             targets = batch[1][j]
#             display_image(image, targets, dataset)
#             if j==4:
#                 break

# Randomly split data into train part and test part
train_data_ratio = 0.95

train_size = int(len(dataset) * train_data_ratio)
test_size = len(dataset) - train_size # test data shouldn't be appeared in training

train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

print("Train data size: ", len(train_data))
print("Test data size: ", len(test_data))


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Define the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 4
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



# Define the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(params, lr=1e-4,
                            amsgrad=True,
                            weight_decay=1e-6)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=5,
                                            gamma=0.5)

# Define the device (GPU or CPU)
device = torch.device('cuda') 

# Define the training loop
def train_one_epoch(model, optimizer, train_dataloader , device, epoch, print_freq):
    model.train()
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        # Sending training data to CUDA
        
        images = list([image.to('cuda') for image in images])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        

        if batch_idx % print_freq == 0:
            print(f'Epoch: {epoch}, Loss: {losses}')
            
            
num_epochs = 20
train_dataloader = DataLoader(train_data, batch_size=32 ,  shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=32 ,  shuffle=True, collate_fn=collate_fn)

# Train the model
model.to(device)
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_dataloader  , device, epoch, print_freq=20)
    lr_scheduler.step()
    
    