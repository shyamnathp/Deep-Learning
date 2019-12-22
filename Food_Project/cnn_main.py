import torch
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.autograd import Variable
from food101 import FOOD101

%load_ext autoreload
%autoreload 2
%matplotlib inline

bs = 64
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Image transformation
# image_transforms = transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

FOOD_PATH = "/home/data/food-101"
IMG_PATH = FOOD_PATH+"/images"
META_PATH = FOOD_PATH+"/meta"
TRAIN_PATH = FOOD_PATH+"/train"
VALID_PATH = FOOD_PATH+"/valid"
MODEL_PATH = 'model_data/'

VGG = 'VGG'
RESNET = 'RESNET'

def parse_args():
    """Add arguments to parser"""
    parser = argparse.ArgumentParser(description='Food Classification Project.')
    parser.add_argument('--model', default=VGG, type=str,
                        choices=[VGG, RESNET], help='initial model for transfer learning')
    parser.add_argument('--batch_number', default=3, type=int,
                        choices=[2, 3], help='number of subsets')
    parser.add_argument('--number_classes', default=10, type=int,
                        choices=[10, 30], help='number of classes')
    args = parser.parse_args()
    return args

def main:
    args = parse_args()
    food = FOOD101()

    # Prepare train dataset by copying images from food-101/images to food-101/train using the file train.txt
    print("Creating train data...")
    prepare_data(META_PATH+'train.txt', IMG_PATH, TRAIN_PATH)

    # Prepare validation data by copying images from food-101/images to food-101/valid using the file test.txt
    print("Creating validation data...")
    prepare_data(META_PATH+'test.txt', IMG_PATH, VALID_PATH)

    print("Total number of samples in train folder")
    !find food-101/train -type d -or -type f -printf '.' | wc -c

    print("Total number of samples in validation folder")
    !find food-101/valid -type d -or -type f -printf '.' | wc -c

    train_ds, valid_ds, classes =  food.get_dataset()
    train_dl, valid_dl = food.get_dls(train_ds, valid_ds, bs=args.batch_number,n=args.number_classes, num_workers=2)

    print("batch size train is", train_dl.batch_size)

    if(args.model == VGG)
        model = model.vgg16(pretrained=True)
    else
        model = model.resnet34(pretrained=True)

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # for idx, data, target in enumerate(train_dl):
    #     torch.save(data, 'data_drive_path{}'.format(idx))
    #     torch.save(target, ...

if __name__ == "__main__":
    main()



