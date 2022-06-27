"""
This is service module!

In here, analyze img sytle!

Logic steps like below!

- init model

- img to tensor

- input file to tensor

- analyze

- serialization
"""
from flask import current_app
import ssl
import torch

from src.utility.load_data import *
from src.utility.ml_gcn import *
from src.utility.engine_test import *


ssl._create_default_https_context = ssl._create_unverified_context


def _init_model(device):
    """
    Title : _init_model
    
    This is def for init model

    Get model state dict from mdel_best.pth.tar and device,

    Return setting model

    Args :
        - device (str) : What computer use for deep learning
    
    Returns :
        - model (object) : Setting model
    
    Raise :
        - Exception : If can'n load model!
    """
    try:
        model = gcn_resnet101(num_classes=10, t=0.03, adj_file="src/data/custom_adj_final.pkl", pretrained=False)

        model.load_state_dict(torch.load("resource/model_best.pth.tar", map_location=device).get("state_dict"))

        model.eval()

        return model

    except Exception as exc:
        current_app.logger.error(f"INIT MODEL ERROR : {exc}")



def _img_to_tensor(device, model, img_path:str):
    """
    Title : _img_to_tensor
    
    This is tranform img to tensor

    Args :
        - device (str) : What computer use for deep learning
        - model (model) : setting model
        - img_path (str) : saved img path
    
    Returns :
        - feature_var (tensor) : transfomred img tenssor
    
    Raise :
        - Exception : If can't transfom img!
    """
    try :
        transforms_test = transforms.Compose([
            Warp(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=model.image_normalization_mean,
                                                     std=model.image_normalization_std),
        ])

        img = Image.open(img_path).convert('RGB')

        image = transforms_test(img).unsqueeze(0).to(device)

        feature_var = torch.autograd.Variable(image).float()
    
        return feature_var

    except Exception as exc:
        current_app.logger.error(f"IMG TO TENSOR ERROR : {exc}")


def _input_to_tensor(device):
    """
    Title : _input_to_tensor
    
    This is tranform input file to tensor

    Args :
        - device (str) : What computer use for deep learning
    
    Returns :
        - inp_var (tensor) : transfomred input file tenssor
    
    Raise :
        - Exception : If can't transfom input file!
    """
    try:
        with open("src/data/custom_glove_word2vec_final.pkl", 'rb') as f:
            inp = pickle.load(f)

        inp_test = torch.Tensor(inp).unsqueeze(0).to(device)

        inp_var = torch.autograd.Variable(inp_test).float().detach()  # one hot

        return inp_var

    except Exception as exc:
        current_app.logger.error(f"INPUT FILE TO TENSOR ERROR : {exc}")


def _output_serialization(output):
    """
    Title : _output_serialization
    
    This is serialization for result to json

    Args :
        - output (list) : style grade list
    
    Returns :
        - result (dict) : dictionary includes "first style" and "second style"
    
    Raise :
        - Exception : If can't serialization!
    """
    try:
        style_value = ["traditional","manish","feminine","ethnic","contemporary","natural","genderless","sporty","subculture","casual"]

        result = zip(style_value, output.detach().numpy()[0])

        result = sorted(result, key= lambda x : -x[1])

        return {"first style" : result[0][0], "second style" : result[1][0]}

    except Exception as exc:
        current_app.logger.error(f"SERIALIZATION ERROR : {exc}")


def style_categorize_service(img_path:str):
    """
    Title : style_categorize_service
    
    This is main def for analyze img sytle!

    Args :
        - img_path (str) : saved img path
    
    Returns :
        - result (dict) : dictionary includes "first style" and "second style"
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    current_app.logger.info(f"\n\
      START ANALIST IMG STYLE \n\
      device : {device} \n\
      img file path : {img_path}")

    model = _init_model(device)

    feature_var = _img_to_tensor(device, model, img_path)

    inp_var = _input_to_tensor(device)

    output = model(feature_var, inp_var)

    result = _output_serialization(output)

    current_app.logger.info(f"\n\
      FINISH ANALIST IMG STYLE \n\
      style : {result}\n\
      img file path : {img_path}")

    return result
