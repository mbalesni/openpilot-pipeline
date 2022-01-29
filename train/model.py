import onnx
from onnx2pytorch import ConvertModel
import torch


def reinitialize_weights(layer_weight):
    torch.nn.init.xavier_uniform_(layer_weight)


def load_model(path_to_supercombo, trainable_layers=[]):

    onnx_model = onnx.load(path_to_supercombo)
    model = ConvertModel(onnx_model, experimental=True)  # pretrained_model

    # enable batch_size > 1 for onnx2pytorch
    model.Constant_1047.constant[0] = -1
    model.Constant_1049.constant[0] = -1
    model.Constant_1051.constant[0] = -1
    model.Constant_1053.constant[0] = -1
    model.Constant_1057.constant[0] = -1
    model.Constant_1059.constant[0] = -1

    # ensure immutability https://github.com/ToriML/onnx2pytorch/pull/38
    model.Elu_907.inplace = False

    # reinitialize trainable layers
    for layer_name, layer in model.named_children():
        # TODO: support layers other than Linear?
        if isinstance(layer, torch.nn.Linear) and layer_name in trainable_layers:
            reinitialize_weights(layer.weight)
            layer.bias.data.fill_(0.01)

    # freeze other layers
    for name, param in model.named_parameters():
        name_layer = name.split(".")[0]
        if name_layer in trainable_layers:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


if __name__ == "__main__":
    pathplan_layer_names  = ["Gemm_959", "Gemm_981","Gemm_983","Gemm_1036"]
    path_to_supercombo = '../common/models/supercombo.onnx'
    model = load_model(pathplan_layer_names, path_to_supercombo)
