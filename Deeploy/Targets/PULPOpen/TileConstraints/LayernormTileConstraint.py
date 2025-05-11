from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class LayernormTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']
        scaleBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        for bufferName in [inputBufferName, outputBufferName, scaleBufferName, biasBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = ctxt.lookup(inputBufferName).shape
        lastDimIdx = len(inputShape) - 1
        lastDimLen = inputShape[-1]

        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == lastDimLen)
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == tilerModel.getTensorDimVar(
                tensorName = scaleBufferName, dimIdx = 0))
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == tilerModel.getTensorDimVar(
                tensorName = biasBufferName, dimIdx = 0))

        for idx, dim in enumerate(inputShape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = outputBufferName, dimIdx = idx))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = ['data_in', 'data_out', 'weight', 'bias']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": []}

        replacementTypes = {"size": PointerClass(uint16_t)}

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            newSize = np.prod(cube.dims)
            replacements["size"].append(newSize)
            weightCube = HyperRectangle((0,), (cube.dims[-1],))
            biasCube = HyperRectangle((0,), (cube.dims[-1],))
            inputLoadSchedule.append({"data_in": cube, "weight": weightCube, "bias": biasCube})
            outputLoadSchedule.append({"data_out": cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule

class LayernormGradTileConstraint(TileConstraint):
    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        grad_in_buffer_name = parseDict['grad_in']          
        data_in_buffer_name = parseDict['data_in']        
        weight_buffer_name = parseDict['weight']          
        bias_buffer_name = parseDict['bias']               
        grad_out_buffer_name = parseDict['grad_out']       
        
        for buffer_name in [grad_in_buffer_name, data_in_buffer_name, weight_buffer_name, 
                           bias_buffer_name, grad_out_buffer_name]:
            tilerModel.addTensorDimToModel(ctxt, buffer_name)
    
        input_shape = ctxt.lookup(data_in_buffer_name).shape
        last_dim_idx = len(input_shape) - 1
        last_dim_len = input_shape[-1]
        

        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=data_in_buffer_name, dimIdx=last_dim_idx) == last_dim_len)
        
  
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=data_in_buffer_name, dimIdx=last_dim_idx) == 
            tilerModel.getTensorDimVar(tensorName=weight_buffer_name, dimIdx=0))
        
    
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName=data_in_buffer_name, dimIdx=last_dim_idx) == 
            tilerModel.getTensorDimVar(tensorName=bias_buffer_name, dimIdx=0))
        
  
        for idx, dim in enumerate(input_shape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=data_in_buffer_name, dimIdx=idx) == 
                tilerModel.getTensorDimVar(tensorName=grad_in_buffer_name, dimIdx=idx))
        
   
        for idx, dim in enumerate(input_shape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName=data_in_buffer_name, dimIdx=idx) == 
                tilerModel.getTensorDimVar(tensorName=grad_out_buffer_name, dimIdx=idx))
        
        
        return tilerModel
    
    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        
        output_cubes = [cube.rectangle for cube in absoluteOutputCubes]
        addr_names = ['grad_in', 'data_in', 'weight', 'bias', 'grad_out']
        input_base_offsets, output_base_offsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addr_names)
        
        replacements = {"size": []}
        replacement_types = {"size": PointerClass(uint16_t)}
        
        input_load_schedule = []
        output_load_schedule = []
        
        for cube in output_cubes:
            new_size = np.prod(cube.dims)
            replacements["size"].append(new_size)
            

            feature_size = cube.dims[-1]
            
            weight_cube = HyperRectangle((0,), (feature_size,))
            bias_cube = HyperRectangle((0,), (feature_size,))
            
            input_load_schedule.append({
                "grad_in": cube,      
                "data_in": cube,     
                "weight": weight_cube,
                "bias": bias_cube
            })
            
            output_load_schedule.append({"grad_out": cube})
        
        tiling_schedule = TilingSchedule(
            input_base_offsets, output_base_offsets, input_load_schedule, output_load_schedule)
        variable_replacement_schedule = VariableReplacementScheme(replacements, replacement_types)
        
        return variable_replacement_schedule, tiling_schedule