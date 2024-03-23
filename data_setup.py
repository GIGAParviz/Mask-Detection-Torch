
from xml.etree import ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import DataLoader , Dataset
import os

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