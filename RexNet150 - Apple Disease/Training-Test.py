import os
import torch
import shutil
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
torch.manual_seed(2024)

class CustomImageDataset(Dataset):
    
    def __init__(self, root_dir, transformations=None):
        
        self.transformations = transformations
        self.image_paths = sorted(glob(f"{root_dir}/*/*"))
        
        self.class_to_idx = {}
        self.class_counts = {}
        count = 0
        
        for img_path in self.image_paths:
            class_name = self._get_class_name(img_path)
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = count
                self.class_counts[class_name] = 1
                count += 1
            else:
                self.class_counts[class_name] += 1
        
    def _get_class_name(self, path):
        return os.path.dirname(path).split("/")[-1]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[self._get_class_name(img_path)]
        
        if self.transformations is not None:
            img = self.transformations(img)
        
        return img, label
    
def create_data_loaders(root_dir, transformations, batch_size, split_ratio=[0.9, 0.05, 0.05], num_workers=4):
    
    dataset = CustomImageDataset(root_dir=root_dir, transformations=transformations)
    
    total_size = len(dataset)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])
    test_size = total_size - (train_size + val_size)
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, dataset.class_to_idx

root_dir = "/kaggle/input/apple-tree-leaf-disease-dataset"
mean, std, image_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
transforms = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])
train_loader, val_loader, test_loader, class_mapping = create_data_loaders(root_dir=root_dir, transformations=transforms, batch_size=32)

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
print(class_mapping)

import random
from matplotlib import pyplot as plt

def tensor_to_image(tensor, color_mode="rgb"):
    
    gray_inverse_transform = T.Compose([
        T.Normalize(mean=[0.], std=[1/0.5]),
        T.Normalize(mean=[-0.5], std=[1])
    ])
    rgb_inverse_transform = T.Compose([
        T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    
    inverse_transform = gray_inverse_transform if color_mode == "gray" else rgb_inverse_transform
    
    return (inverse_transform(tensor) * 255).detach().squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)

def plot_samples(data_loader, num_images, rows, color_map=None, class_names=None):
    
    assert color_map in ["rgb", "gray"], "Specify color_map as either 'rgb' or 'gray'!"
    if color_map == "rgb": color_map = "viridis"
    
    plt.figure(figsize=(20, 10))
    indices = [random.randint(0, len(data_loader.dataset) - 1) for _ in range(num_images)]
    for idx, index in enumerate(indices):
        
        img, label = data_loader.dataset[index]
        plt.subplot(rows, num_images // rows, idx + 1)
        plt.imshow(tensor_to_image(img, color_map), cmap=color_map)
        plt.axis('off')
        if class_names is not None:
            plt.title(f"GT -> {class_names[int(label)]}")
        else:
            plt.title(f"GT -> {label}")
            
plot_samples(train_loader, 20, 4, "rgb", list(class_mapping.keys()))
plot_samples(val_loader, 20, 4, "rgb", list(class_mapping.keys()))
plot_samples(test_loader, 20, 4, "rgb", list(class_mapping.keys()))

def analyze_data(root_dir, transformations):
    
    dataset = CustomImageDataset(root_dir=root_dir, transformations=transformations)
    class_counts = dataset.class_counts
    bar_width = 0.7
    text_width = 0.05
    text_height = 2
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    
    fig, ax = plt.subplots(figsize=(20, 10))
    indices = np.arange(len(counts))

    ax.bar(indices, counts, bar_width, color="firebrick")
    ax.set_xlabel("Class Names", color="red")
    ax.set_xticklabels(class_names, rotation=60)
    ax.set(xticks=indices, xticklabels=class_names)
    ax.set_ylabel("Data Counts", color="red")
    ax.set_title("Dataset Class Imbalance Analysis")

    for i, count in enumerate(counts):
        ax.text(i - text_width, count + text_height, str(count), color="royalblue")
    
analyze_data(root_dir=root_dir, transformations=transforms)

import timm
from tqdm import tqdm

model = timm.create_model("rexnet_150", pretrained=True, num_classes=len(class_mapping))

def setup_training(model):
    return model.to("cuda").eval(), 10, "cuda", torch.nn.CrossEntropyLoss(), torch.optim.Adam(params=model.parameters(), lr=3e-4)

def move_batch_to_device(batch, device):
    return batch[0].to(device), batch[1].to(device)

def calculate_metrics(model, images, labels, loss_fn, epoch_loss, epoch_accuracy):
    predictions = model(images)
    loss = loss_fn(predictions, labels)
    return loss, epoch_loss + (loss.item()), epoch_accuracy + (torch.argmax(predictions, dim=1) == labels).sum().item()

model, num_epochs, device, loss_fn, optimizer = setup_training(model)

save_prefix, save_directory = "disease", "saved_models"
print("Start training...")
best_accuracy, best_loss, threshold, no_improvement, patience = 0, float("inf"), 0.01, 0, 5
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

best_loss = float(torch.inf)
    
for epoch in range(num_epochs):

    epoch_loss, epoch_accuracy = 0, 0
    for idx, batch in tqdm(enumerate(train_loader)):

        images, labels = move_batch_to_device(batch, device)

        loss, epoch_loss, epoch_accuracy = calculate_metrics(model, images, labels, loss_fn, epoch_loss, epoch_accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_accuracy = epoch_accuracy / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    print(f"{epoch + 1}-epoch train process is completed!")
    print(f"{epoch + 1}-epoch train loss          -> {avg_train_loss:.3f}")
    print(f"{epoch + 1}-epoch train accuracy      -> {avg_train_accuracy:.3f}")

    model.eval()
    with torch.no_grad():
        val_epoch_loss, val_epoch_accuracy = 0, 0
        for idx, batch in enumerate(val_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, labels)
            pred_classes = torch.argmax(predictions.data, dim=1)
            val_epoch_accuracy += (pred_classes == labels).sum().item()
            val_epoch_loss += loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        avg_val_accuracy = val_epoch_accuracy / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        print(f"{epoch + 1}-epoch validation process is completed!")
        print(f"{epoch + 1}-epoch validation loss     -> {avg_val_loss:.3f}")
        print(f"{epoch + 1}-epoch validation accuracy -> {avg_val_accuracy:.3f}")

        if avg_val_loss < (best_loss + threshold):
            os.makedirs(save_directory, exist_ok=True)
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"{save_directory}/{save_prefix}_best_model.pth")
            
        else:
            no_improvement += 1
            print(f"Loss value did not decrease for {no_improvement} epochs")
            if no_improvement == patience:
                print(f"Stop training since loss value did not decrease for {patience} epochs.")
                break

def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", c="red")
    plt.plot(val_losses, label="Validation Loss", c="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Values")
    plt.xticks(ticks=np.arange(len(train_losses)), labels=[i for i in range(1, len(train_losses) + 1)])
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy", c="orangered")
    plt.plot(val_accuracies, label="Validation Accuracy", c="darkgreen")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Scores")
    plt.xticks(ticks=np.arange(len(train_accuracies)), labels=[i for i in range(1, len(train_accuracies) + 1)])
    plt.legend()
    plt.show()
    
plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)

import cv2

class FeatureSaver:
    
    """ Extract pretrained activations"""
    features = None
    def __init__(self, model):
        self.hook = model.register_forward_hook(self._hook_function)
    def _hook_function(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()

def compute_CAM(conv_features, linear_weights, class_idx):
    
    bs, chs, h, w = conv_features.shape
    cam = linear_weights[class_idx].dot(conv_features[0, :, :, ].reshape((chs, h * w)))
    cam = cam.reshape(h, w)
    
    return (cam - np.min(cam)) / np.max(cam)

def run_inference(model, device, test_loader, num_images, rows, final_conv_layer, fc_params, class_names=None):
    
    weights, accuracy = np.squeeze(fc_params[0].cpu().data.numpy()), 0
    feature_saver = FeatureSaver(final_conv_layer)
    predictions, image_list, labels_list = [], [], []
    
    for idx, batch in tqdm(enumerate(test_loader)):
        images, labels = move_batch_to_device(batch, device)
        pred_classes = torch.argmax(model(images), dim=1)
        accuracy += (pred_classes == labels).sum().item()
        image_list.append(images)
        predictions.append(pred_classes.item())
        labels_list.append(labels.item())
    
    print(f"Accuracy of the model on the test data -> {(accuracy / len(test_loader.dataset)):.3f}")
    
    plt.figure(figsize=(20, 10))
    indices = [random.randint(0, len(image_list) - 1) for _ in range(num_images)]
    
    for idx, index in enumerate(indices):
        
        img = image_list[index].squeeze()
        pred_class_idx = predictions[index]
        heatmap = compute_CAM(feature_saver.features, weights, pred_class_idx)
        
        plt.subplot(rows, num_images // rows, idx + 1)
        plt.imshow(tensor_to_image(img), cmap="gray")
        plt.axis("off")
        plt.imshow(cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR), alpha=0.4, cmap='jet')
        plt.axis("off")
        
        if class_names is not None:
            plt.title(f"GT -> {class_names[int(labels_list[index])]} ; PRED -> {class_names[int(predictions[index])]}", color=("green" if class_names[int(labels_list[index])] == class_names[int(predictions[index])] else "red"))
        else:
            plt.title(f"GT -> {labels_list[index]} ; PRED -> {predictions[index]}")

model.load_state_dict(torch.load(f"{save_directory}/{save_prefix}_best_model.pth"))
model.eval()
final_conv_layer, fc_parameters = model.features[-1], list(model.head.fc.parameters())
run_inference(model=model.to(device), device=device, test_loader=test_loader, num_images=20, rows=4, class_names=list(class_mapping.keys()), final_conv_layer=final_conv_layer, fc_params=fc_parameters)