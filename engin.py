import torch
from tqdm import tqdm
from torch.utils.data import DataLoader , Dataset


def train_step(model: torch.nn.Module,
               train_dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               print_freq:int):
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=5,
                                                gamma=0.5)

        
    for batch , (images , targets) in tqdm(enumerate(train_dataloader)):
        
        images = list([image.to('cuda') for image in images])
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        train_loss = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        
        train_loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        
        if batch % print_freq == 0:
            print(f'Train Loss: {train_loss}')
            
        return train_loss

def test_step(model: torch.nn.Module,
               test_dataloader: DataLoader,
               device: torch.device):

        
    for (images , targets) in test_dataloader:
        
        images = list([image.to('cuda') for image in images])
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        model.eval()
        with torch.no_grad():
            
            loss_dict = model(images, targets)
            
            test_loss = sum(loss for loss in loss_dict.values())

            return test_loss