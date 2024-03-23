
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import patches
import torchvision.ops as ops
import torch 
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def display_image(image, targets):
    fig, ax = plt.subplots(1)
    unnormalized_image = unorm(image).cpu().permute(1, 2, 0).numpy()
    ax.imshow(unnormalized_image)
    boxes = targets['boxes']
    labels = targets['labels']
    label_color = ['green', 'red', 'purple' ]
    for j, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin),( xmax - xmin), (ymax - ymin), linewidth=2, edgecolor=label_color[labels[j] - 1], facecolor='none')
        ax.add_patch(rect)
    plt.show()


def show_imgs(dataloader):
    for i, batch in enumerate(dataloader):
        if i == 3:
            for j in range(len(batch[0])):
                image = batch[0][j]
                targets = batch[1][j]
                display_image(image, targets)
                if j==4:
                    break

def test_model(model: torch.nn.Module,
               test_dataloader:DataLoader,
               device: torch.device
               ):

    images, targets = next(iter(test_dataloader))

    # Run the model on the minibatch of images
    model.eval()
    with torch.no_grad():
        images = list([image.to(device) for image in images])
        outputs = model(images)

        for i, image in enumerate(images):
            # Extract the predicted bounding boxes and labels for the current image
            boxes = outputs[i]['boxes'].cpu().numpy()
    #         scores = outputs[i]['scores'].cpu().numpy()
            labels = outputs[i]['labels'].cpu().numpy()
            
            # Apply non-maximum suppression to remove duplicate detections
    #         keep = ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold=0.1)

            # Keep only the top-scoring detections after NMS
    #         boxes = boxes[keep]
    # #         print(boxes.shape)
    #         scores = scores[keep]
    #         labels = labels[keep]

            fig, ax = plt.subplots(1)
            ax.imshow(unorm(image).cpu().permute(1, 2, 0))
            label_color = ['green', 'red', 'purple' ]

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.astype(np.int)
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=label_color[label - 1], linewidth=2))

            plt.show()
