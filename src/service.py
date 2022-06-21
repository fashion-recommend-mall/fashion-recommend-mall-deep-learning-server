import torch
from src.utility.load_data import *
from src.utility.ml_gcn import *
from src.utility.engine_test import *
import ssl

def _init_model(device):

    ssl._create_default_https_context = ssl._create_unverified_context

    model = gcn_resnet101(num_classes=10, t=0.03, adj_file="src/data/custom_adj_final.pkl", pretrained=False)

    model.load_state_dict(torch.load("model_best.pth.tar", map_location=device).get("state_dict"))

    model.eval()
    
    return model

def _img_to_tensor(device, model, img_path:str):

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

def _input_to_tensor(device):

    with open("src/data/custom_glove_word2vec_final.pkl", 'rb') as f:
        inp = pickle.load(f)

    inp_test = torch.Tensor(inp).unsqueeze(0).to(device)

    inp_var = torch.autograd.Variable(inp_test).float().detach()  # one hot

    return inp_var


def _output_serialization(output):
    
    style_value = ["traditional","manish","feminine","ethnic","contemporary","natural","genderless","sporty","subculture","casual"]

    result = zip(style_value, output.detach().numpy()[0])

    result = sorted(result, key= lambda x : -x[1])

    return {"first style" : result[0][0], "second style" : result[1][0]}

def style_categorize_service(img_path:str):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _init_model(device)

    feature_var = _img_to_tensor(device, model, img_path)

    inp_var = _input_to_tensor(device)

    output = model(feature_var, inp_var)

    result = _output_serialization(output)

    return result

    
