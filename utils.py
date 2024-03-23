
from matplotlib import pyplot as plt
from matplotlib import patches

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