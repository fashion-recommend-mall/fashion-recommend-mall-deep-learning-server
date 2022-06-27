from flask import current_app
import ssl
import torch

from src.utility.load_data import *
from src.utility.ml_gcn import *
from src.utility.engine_test import *


ssl._create_default_https_context = ssl._create_unverified_context


def _init_model(device):
    try:
        model = gcn_resnet101(num_classes=10, t=0.03, adj_file="src/data/custom_adj_final.pkl", pretrained=False)

        model.load_state_dict(torch.load("resource/model_best.pth.tar", map_location=device).get("state_dict"))

        model.eval()

        return model

    except Exception as exc:
        current_app.logger.error(f"INIT MODEL ERROR : {exc}")



def _img_to_tensor(device, model, img_path:str):
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
    try:
        with open("src/data/custom_glove_word2vec_final.pkl", 'rb') as f:
            inp = pickle.load(f)

        inp_test = torch.Tensor(inp).unsqueeze(0).to(device)

        inp_var = torch.autograd.Variable(inp_test).float().detach()  # one hot

        return inp_var

    except Exception as exc:
        current_app.logger.error(f"INPUT FILE TO TENSOR ERROR : {exc}")


def _output_serialization(output):
    try:
        style_value = ["traditional","manish","feminine","ethnic","contemporary","natural","genderless","sporty","subculture","casual"]

        result = zip(style_value, output.detach().numpy()[0])

        result = sorted(result, key= lambda x : -x[1])

        return {"first style" : result[0][0], "second style" : result[1][0]}

    except Exception as exc:
        current_app.logger.error(f"SERIALIZATION ERROR : {exc}")


def style_categorize_service(img_path:str):

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
