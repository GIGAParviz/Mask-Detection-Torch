# import warnings
# warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from data_setup import DatasetMaker
from model_builder import build_fasterrcnn_model
from engin import train_step , test_step 
from utils import show_imgs
from colorama import Fore
from utils import test_model

root_dir = r"D:\deep_learning\Torch\Mask Detection\face-mask-detection"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485 , 0.456 , 0.406] , std=[0.229, 0.224, 0.225])
])

collate_fn = lambda batch: tuple(zip(*batch))

dataset = DatasetMaker(root_dir=root_dir , transform=transform)
dataloader = DataLoader(dataset ,  shuffle=True, collate_fn=collate_fn , batch_size=32)

train_size = int(len(dataset) * 0.95)
test_size = len(dataset) - train_size

train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

print("Train data size: ", len(train_data))
print("Test data size: ", len(test_data))

train_dataloader = DataLoader(train_data, batch_size=32 ,  shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=32 ,  shuffle=True, collate_fn=collate_fn)

# show_imgs(train_dataloader)

device = torch.device('cuda') 

model = build_fasterrcnn_model(device)

params = model.parameters()

optimizer = torch.optim.AdamW(params, lr=1e-4,
                            amsgrad=True,
                            weight_decay=1e-6)

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          print_freq:int):
    
    
    for epoch in range(epochs):
        
        train_loss = train_step(model,
                                train_dataloader,
                                optimizer,
                                device,
                                print_freq
                                )
        test_loss = test_step(model,
                              test_dataloader,
                              device)
            
        print(
            Fore.YELLOW + F"Epoch: {epoch}" ,
            Fore.GREEN +f"|train loss: {train_loss:.4f} | ",
            Fore.LIGHTRED_EX +f"|test loss: {test_loss:.4f}"
            )
            
            
        resutls = {"train_loss" : [],
                   "test loss" : []}
        
        resutls["train_loss"].append(train_loss)                      
        resutls["test_loss"].append(test_loss)
        
epochs = 10      
print_freq = 10

def main():
    train(model,
          train_dataloader,
          test_dataloader,
          optimizer,
          epochs,
          device,
          print_freq)
    
    test_model(model,
           test_dataloader,
           device)
if __name__ == "__main__":
    main()
