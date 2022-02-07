import onnx
from onnx2pytorch import ConvertModel
import torch
import onnxruntime as rt
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_MODEL = os.path.join(parent_dir, 'common/models/supercombo.onnx')


def reinitialize_weights(layer_weight):
    torch.nn.init.xavier_uniform_(layer_weight)


def load_trainable_model(path_to_supercombo, trainable_layers=[]):

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


def load_inference_model(path_to_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if path_to_model.endswith('.onnx'):
        onnx_graph = onnx.load(path_to_model)
        output_names = [node.name for node in onnx_graph.graph.output]
        model = rt.InferenceSession(path_to_model, providers=['CPUExecutionProvider'])

        def run_model(inputs):
            outs =  model.run(output_names, inputs)[0]
            recurrent_state = outs[:, -512:]
            return outs, recurrent_state


    elif path_to_model.endswith('.pth'):

        model = load_trainable_model(ORIGINAL_MODEL)
        model.load_state_dict(torch.load(path_to_model))
        model.eval()
        model = model.to(device)

        def run_model(inputs):
            with torch.no_grad():
                inputs = {k: torch.from_numpy(v).to(device) for k, v in inputs.items()}
                outs = model(**inputs)
                recurrent_state = outs[:, -512:]
                return outs.cpu().numpy(), recurrent_state

    return model, run_model


if __name__ == "__main__":
    pathplan_layer_names  = ["Gemm_959", "Gemm_981","Gemm_983","Gemm_1036"]
    path_to_supercombo = '../common/models/supercombo.onnx'
    model = load_trainable_model(pathplan_layer_names, path_to_supercombo)
