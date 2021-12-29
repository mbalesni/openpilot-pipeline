from onnx2pytorch import ConvertModel
from tensorflow.keras.layers import BatchNormalization
import time
import numpy as np
import torch
import onnxruntime as rt
import onnx
from onnx2keras import onnx_to_keras
from keract import get_activations
import colorama
import matplotlib.pyplot as plt

import os


def printf(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def color_thresh(x, threshold=1e-3):
    c = colorama.Fore.RED if x >= threshold else colorama.Fore.WHITE
    return f'{c}{x}'


PATH = 4955
LANE_LINES = PATH+528
LANE_LINE_PROB = LANE_LINES+8
ROAD_EDGES = LANE_LINE_PROB+264
LEADS = ROAD_EDGES+102
LEAD_PROB = LEADS+3
DESIRE_STATE = LEAD_PROB+8
META = DESIRE_STATE+80
POSE = META+12
RECURRENT_STATE = POSE+512

zone_starts = [PATH, LANE_LINES, LANE_LINE_PROB, ROAD_EDGES, LEADS, LEAD_PROB, DESIRE_STATE, META, POSE, RECURRENT_STATE]
zone_names_all_caps = ['PATH', 'LANE_LINES', 'LANE_LINE_PROB', 'ROAD_EDGES', 'LEADS', 'LEAD_PROB', 'DESIRE_STATE', 'META', 'POSE', 'RECURRENT_STATE']
zone_names_capital = [x.capitalize() for x in zone_names_all_caps]



def get_zone_index(x_index):
	for i, zone_start in enumerate(zone_starts):
		if x_index < zone_start:
			return i


zone_indices = [get_zone_index(i) for i in range(RECURRENT_STATE)]


def extract_best_path(outs):
	outs = outs[0]
	path_plans = outs[:PATH]

	print('path plans', path_plans.shape)

	paths = np.array(np.split(path_plans, 5, axis=0))
	paths = paths.squeeze() # (5, 991)

	best_idx = np.argmax(paths[:, -1], axis=0)
	best_path = paths[best_idx, :-1].reshape(2, 33, 15)

	return best_path

if __name__ == '__main__':

	path_to_onnx_model = 'train/supercombo.onnx'

	# INIT

	# onnx
	model = onnx.load(path_to_onnx_model)

	input_names = [node.name for node in model.graph.input]
	output_names = [node.name for node in model.graph.output]

	printf('Inputs: ', input_names)
	printf('Outputs: ', output_names)

	# onnxruntime
	providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
	# providers = ["CPUExecutionProvider"]
	onnxruntime_model = rt.InferenceSession(path_to_onnx_model, providers=providers)

	# pytorch
	device = torch.device('cuda')
	pytorch_model = ConvertModel(model, experimental=False, debug=False).to(device)
	pytorch_model.train(False)  # NOTE: important

	# pytorch_model.Constant_1047.constant= torch.tensor((2,2,66))
	# pytorch_model.Reshape_1048.shape = (2,2,66)
	# pytorch_model.Constant_1049.constant = torch.tensor((2,2,66))
	# pytorch_model.Reshape_1050.shape = (2,2,66)
	# pytorch_model.Constant_1051.constant = torch.tensor((2,2,66))
	# pytorch_model.Reshape_1052.shape = (2,2,66)
	# pytorch_model.Constant_1053.constant = torch.tensor((2,2,66))
	# pytorch_model.Reshape_1054.shape = (2,2,66)
	# pytorch_model.Constant_1057.constant = torch.tensor((2,2,66))
	# pytorch_model.Reshape_1058.shape = (2,2,66)
	# pytorch_model.Constant_1059.constant = torch.tensor((2,2,66))
	# pytorch_model.Reshape_1060.shape = (2,2,66)

	# keras
	keras_model = onnx_to_keras(model, input_names, verbose=False)

	# INFERENCE

	# comma_recordings_basedir = '/home/nikita/data'
	# train_split = 0.8
	# seq_len = 2
	# single_frame_batches = False
	# batch_size = 1
	# num_workers = 1
	# prefetch_factor = 1


	# dummy inputs
	inputs = {
		'input_imgs': np.ones((1, 12, 128, 256), dtype=np.float32),
		'desire': np.ones((1, 8), dtype=np.float32),
		'traffic_convention': np.ones((1, 2), dtype=np.float32),
		'initial_state': np.ones((1, 512), dtype=np.float32)
	}

	torch_inputs = {k: torch.from_numpy(v).to(device) for k, v in inputs.items()}

	layer_names = [layer.name for layer in keras_model.layers]
	layer_outs = get_activations(keras_model, list(inputs.values()), layer_names=None, output_format='simple', auto_compile=True)
	layer_outs = {k: v for k, v in layer_outs.items() if k not in input_names}
	print('layer outs:', list(layer_outs.keys()))

	outs_keras = keras_model(inputs)
	outs_torch = pytorch_model(**torch_inputs, expected_activations=layer_outs)
	outs_onnx = onnxruntime_model.run(output_names, inputs)[0]

	# printf(keras_model.summary())
	# printf('outs keras:', outs_keras.shape)

	try:
		outs_torch_np = outs_torch.detach().cpu().numpy()
	except Exception as err:
		printf(err)


	# compare
	# diff_torch_keras = np.max(outs_torch_np - outs_keras)
	# diff_keras_onnx = np.max(outs_keras - outs_onnx)
	# diff_torch_onnx = np.max(np.abs(outs_torch_np - outs_onnx))

	# print each diff
	# printf('max diff torch-keras', diff_torch_keras)
	# printf('max diff keras-onnx', diff_keras_onnx)
	# printf('max diff torch-onnx', diff_torch_onnx)


	# keras_path = extract_best_path(outs_keras)
	# pytorch_path = extract_best_path(outs_onnx)

	# path_diff = np.abs(keras_path - pytorch_path)
	# significant_diff_mask = path_diff > 0.05

	# print keras and pytorch path with differences in red
	torch.set_printoptions(profile="full")
	np.set_printoptions(formatter={'float': color_thresh}, threshold=10000, linewidth=250)
	print('diff:')
	# print(path_diff)
	diff_keras = np.abs(outs_torch_np[0] - outs_keras[0])
	diff_onnxruntime = np.abs(outs_torch_np[0] - outs_onnx[0])

	xticks = [0]
	xticks += list(np.arange(0,6) * PATH/5)
	xticks += [POSE, RECURRENT_STATE]


	plt.figure(figsize=(20, 10))

	plt.title('PyTorch Model Output Error')
	plt.ylabel('Absolute diff between PyTorch & Keras (log)')
	plt.xlabel('Output position')
	plt.yscale('log')
	plt.xticks(xticks)

	start = 0
	for i, zone_start in enumerate(zone_starts):
		end = zone_start
		plt.scatter(np.arange(start, end), diff_keras[start:end], alpha=0.3, label=f'{zone_names_capital[i]}')
		start = end

	plt.legend()
	plt.savefig('diff_keras.png')
	plt.close()


	# onnxruntime

	plt.figure(figsize=(20, 10))

	plt.title('PyTorch Model Output Error')
	plt.ylabel('Absolute diff between PyTorch & ONNXRuntime (log)')
	plt.xlabel('Output position')
	plt.yscale('log')
	plt.xticks(xticks)

	start = 0
	for i, zone_start in enumerate(zone_starts):
		end = zone_start
		plt.scatter(np.arange(start, end), diff_onnxruntime[start:end], alpha=0.3, label=f'{zone_names_capital[i]}')
		start = end

	plt.legend()
	plt.savefig('diff_onnxruntime.png')





	printf('DONE')
