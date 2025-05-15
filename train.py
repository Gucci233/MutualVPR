import os
import random
import sys
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
import torch.nn as nn
import test
import util
import myparser
import commons
import cosface_loss
import augmentations
from cosplace_model import cosplace_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
import numpy as np
#from sklearn.cluster import KMeans
torch.backends.cudnn.benchmark = True  # Provides a speedup
import concurrent.futures
from sklearn.cluster import KMeans

args = myparser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed+2015)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Model
if args.backbone =='DinoV2':
    model = cosplace_network.MutualVPR(pretrained_foundation = True, foundation_model_path = '/your/path/dinov2_vitb14_pretrain.pth',output_dim=args.fc_output_dim)
else:
    model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.train_all_layers)
logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")



if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = nn.DataParallel(model)
model = model.to(args.device)

if args.backbone =='DinoV2':
    ## Freeze parameters except adapter
    for name, param in model.module.backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False

    ## initialize Adapter
    for n, m in model.named_modules():
        if 'adapter' in n:
            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.)
                        nn.init.constant_(m2.bias, 0.)
            for n2, m2 in m.named_modules():
                if 'conv' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)
#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=50)

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, N=args.N, C=args.C,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]


logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold,
                     image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                      positive_dist_threshold=25,
                      image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)

logging.info(f"Validation set: {val_ds}")
# logging.info(f"Test set: {test_ds}")
#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers,)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0


#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size)

if args.augmentation_device == "cuda":
    gpu_augmentation = T.Compose([
            augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue),
            augmentations.DeviceAgnosticRandomResizedCrop([args.image_size, args.image_size],
                                                          scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

test_transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

random_transform = T.Compose([
    augmentations.DeviceAgnosticRandomResizedCrop([args.image_size, args.image_size],
                                                scale=[1-args.random_resized_crop, 1]),
    augmentations.DeviceAgnosticRandomHorizontalFlip(),
    augmentations.DeviceAgnosticColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




import concurrent.futures

def get_images(key):
    images_path = groups[current_group_num].image_dict[key]
    images_tensor = torch.empty(len(images_path), 3, args.image_size, args.image_size)
    
    for i, path in enumerate(images_path):
        pil_image = TrainDataset.open_image(path) 
        images_tensor[i] = test_transform(pil_image)  

    return images_tensor

images_dict = {}
def load_images_concurrently(keys):
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {key: executor.submit(get_images, key) for key in keys}
        
        for future in concurrent.futures.as_completed(futures.values()):
            key = next(key for key, fut in futures.items() if fut == future)
            result = future.result() 

            images_dict[key] = result 


def get_labels(images_tensor):
    images = images_tensor.to(args.device)


    total_samples = images.size(0)
    descriptors = np.empty((total_samples, args.fc_output_dim), dtype="float32")
    bs = 128
    with torch.no_grad():
        for i in range(0, total_samples, bs):
            start_idx = i
            end_idx = min(i + bs, total_samples)
            batch_tensor = images[start_idx:end_idx]
            bs_descriptors = model(batch_tensor)
            bs_descriptors = bs_descriptors.cpu().numpy()
            descriptors[start_idx:end_idx] = bs_descriptors

        del images

    
    kmeans = KMeans(n_clusters=args.C, n_init=10).fit(descriptors)
    P = kmeans.labels_
    return P




for epoch_num in range(start_epoch_num, args.epochs_num):
    
    #### Train
    epoch_start_time = datetime.now()

    current_group_num = epoch_num % args.groups_num
    current_group_num0 = 0
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)

    
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
        
    samples = random.sample(list(groups[current_group_num].image_dict.keys()), len(groups[current_group_num].image_dict) // 5)

    for keys in tqdm(range(0,len(samples),30)):
        load_images_concurrently(samples[keys:keys+30])
        for key in tqdm(images_dict.keys()):
            labels_list = get_labels(images_dict[key])
            if len(set(labels_list)) < args.C:
                continue
            groups[current_group_num].update_classes(labels_list,key)
        images_dict = {}
        
    
    dataloader = commons.InfiniteDataLoader(groups[current_group_num],
                                            batch_size=args.batch_size, shuffle=True,num_workers=8,
                                            pin_memory=True, drop_last=True)
    
    dataloader_iterator = iter(dataloader)
    model = model.train()
    for iteration in tqdm(iterable=range(args.iterations_per_epoch), ncols=100):
        images, targets,indices= next(dataloader_iterator)
        images, targets = images.to(args.device), targets.to(args.device)
        
        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)


        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()

        descriptors = model(images)
        output = classifiers[current_group_num](descriptors, targets)
        loss = criterion(output, targets)
        loss.backward()
        epoch_losses = np.append(epoch_losses, loss.item())
        model_optimizer.step()
        classifiers_optimizers[current_group_num].step()
        del images,loss, output,targets
    
    # scheduler.step()
    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")


    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}")

    #### Evaluation
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, args.output_folder)
    
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")