# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float Layernorm (Name: ${nodeName}, Op: ${nodeOp})
// Outputs: Y (data_out), mean stash, inv_std_dev stash
PULP_Layernorm_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
    ${data_in},
    ${data_out},
    ${weight},
    ${bias},
    ${size},
    ${lastDimLength},
    ${epsilon}
);
""")

referenceGradTemplate = NodeTemplate("""
// FloatLayernormGrad Parallel (Name: ${nodeName}, Op: ${nodeOp})
// Uses pre-computed mean/inv_std_dev stash from forward pass

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);

int32_t ${nodeName}_seq_length = ${size} / ${lastDimLength};
int32_t ${nodeName}_chunk = (${nodeName}_seq_length >> ${nodeName}_log2Core) +
                          ((${nodeName}_seq_length & (NUM_CORES-1)) != 0);
int32_t ${nodeName}_start = MIN(${nodeName}_chunk * ${nodeName}_core_id, ${nodeName}_seq_length);
int32_t ${nodeName}_end = MIN(${nodeName}_start + ${nodeName}_chunk, ${nodeName}_seq_length);

int32_t ${nodeName}_elem_start = ${nodeName}_start * ${lastDimLength};
int32_t ${nodeName}_elem_end = ${nodeName}_end * ${lastDimLength};
int32_t ${nodeName}_elem_count = ${nodeName}_elem_end - ${nodeName}_elem_start;

const float${grad_in_type.referencedType.typeWidth}_t* ${nodeName}_grad_in_ptr = ${grad_in} + ${nodeName}_elem_start;
const float${data_in_type.referencedType.typeWidth}_t* ${nodeName}_data_in_ptr = ${data_in} + ${nodeName}_elem_start;
float${grad_out_type.referencedType.typeWidth}_t* ${nodeName}_grad_out_ptr = ${grad_out} + ${nodeName}_elem_start;
const float${mean_type.referencedType.typeWidth}_t* ${nodeName}_mean_ptr = ${mean} + ${nodeName}_start;
const float${inv_std_dev_type.referencedType.typeWidth}_t* ${nodeName}_inv_std_dev_ptr = ${inv_std_dev} + ${nodeName}_start;

// Parallel: compute dX for each core's chunk of sequences using stash
if (${nodeName}_elem_count > 0) {
  PULP_LayernormGrad_fp${grad_in_type.referencedType.typeWidth}_fp${grad_out_type.referencedType.typeWidth}(
      ${nodeName}_grad_in_ptr,         // Upstream gradient (dY) - chunk
      ${nodeName}_data_in_ptr,         // Original input (X) - chunk
      ${nodeName}_mean_ptr,            // Stash mean - chunk
      ${nodeName}_inv_std_dev_ptr,     // Stash inv_std_dev - chunk
      ${nodeName}_grad_out_ptr,        // Output gradient (dX) - chunk
      ${weight},                       // Scale parameter (gamma)
      ${nodeName}_elem_count,          // Number of elements to process
      ${lastDimLength}                 // Size of the feature dimension
  );
}

// Core 0 only: compute dscale (weight_grad) and dbias (bias_grad) over all sequences
if (${nodeName}_core_id == 0) {
  PULP_LayernormGradParam_fp${grad_in_type.referencedType.typeWidth}_fp${grad_out_type.referencedType.typeWidth}(
      ${grad_in},                      // Full upstream gradient (dY)
      ${data_in},                      // Full original input (X)
      ${mean},                         // Stash mean (all sequences)
      ${inv_std_dev},                  // Stash inv_std_dev (all sequences)
      ${weight_grad},                  // Output: gradient w.r.t. scale (dscale)
      ${bias_grad},                    // Output: gradient w.r.t. bias (dbias)
      ${size},                         // Total number of elements
      ${lastDimLength}                 // Size of the feature dimension
  );
}
""")
