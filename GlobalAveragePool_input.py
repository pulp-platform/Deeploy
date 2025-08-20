import numpy as np

# random input data
input_data = np.random.randn(2, 3, 20, 40).astype(np.float32)
expected_output = np.mean(input_data, axis=(2, 3)).reshape(2, 3, 1, 1)


np.savez('/app/deeploy/DeeployTest/Tests/GlobalAveragePool/inputs.npz', 
         input_0=input_data, output_0=expected_output)
