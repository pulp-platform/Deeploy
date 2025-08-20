import numpy as np
import onnx
import onnxruntime as ort

def generate_output_npz():

    model_path = "/app/deeploy/DeeployTest/Tests/GlobalAveragePool/network.onnx"  # 或 "attached_model.onnx"
    inputs_npz_path = "/app/deeploy/DeeployTest/Tests/GlobalAveragePool/inputs.npz"
    output_npz_path = "/app/deeploy/DeeployTest/Tests/GlobalAveragePool/outputs.npz"

   
    inputs = np.load(inputs_npz_path)
    input_data = inputs['input_0']  

   
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    expected_output = sess.run(None, {input_name: input_data})[0]


    np.savez(inputs_npz_path, input_0=input_data)
    np.savez(output_npz_path, output_0=expected_output)
    print(f"Generated {inputs_npz_path} with expected output shape: {input_data.shape}, dtype: {input_data.dtype}")
    
    print(f"Generated output.npz with keys: {list(np.load(output_npz_path).keys())}")
    print(f"Output shape: {np.load(output_npz_path)['output_0'].shape}, dtype: {np.load(output_npz_path)['output_0'].dtype}")

if __name__ == "__main__":
    generate_output_npz()