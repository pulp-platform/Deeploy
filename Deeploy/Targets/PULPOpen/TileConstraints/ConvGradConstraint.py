# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class ConvGradX2DTileConstraint(TileConstraint):
    """
    ConvGradX (dX) TileConstraint for your naive trainlib kernel (NCHW),
    with weight layout: W = [C_out, C_in_per_group, P, Q].

    Tensor mapping (matches your NodeTemplate):
      data_in  = dY = gradOut  (smaller)  [N, C_out, H_out, W_out]  NCHW
      data_out = dX = gradIn   (larger)   [N, C_in,  H_in,  W_in ]  NCHW
      weight   = W                          [C_out, C_in_per_group, P, Q]

    Kernel behavior:
      - memset(pGradIn, 0, C_in*H_in*W_in) inside each call
      => Must NOT tile C_out (would require accumulation into dX)
      => Must NOT tile C_in  (same)
      => Only tile N and/or H_in/W_in.
    """

    # ---------------------------
    # 1) Geometrical constraints
    # ---------------------------
    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        gradOutName = parseDict["data_in"]    # dY
        gradInName  = parseDict["data_out"]   # dX
        weightName  = parseDict["weight"]

        tilerModel.addTensorDimToModel(ctxt, gradOutName)
        tilerModel.addTensorDimToModel(ctxt, gradInName)
        tilerModel.addTensorDimToModel(ctxt, weightName)

        pads    = parseDict["pads"]       # [pad_top, pad_bottom, pad_left, pad_right]
        strides = parseDict["strides"]    # [stride_h, stride_w]
        group   = parseDict["group"]

        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides

        # NCHW dims
        N_go  = tilerModel.getTensorDimVar(gradOutName, 0)
        Co_go = tilerModel.getTensorDimVar(gradOutName, 1)
        Ho_go = tilerModel.getTensorDimVar(gradOutName, 2)
        Wo_go = tilerModel.getTensorDimVar(gradOutName, 3)

        N_gi  = tilerModel.getTensorDimVar(gradInName, 0)
        Ci_gi = tilerModel.getTensorDimVar(gradInName, 1)
        Hi_gi = tilerModel.getTensorDimVar(gradInName, 2)
        Wi_gi = tilerModel.getTensorDimVar(gradInName, 3)

        # weight dims: [C_out, C_in_per_group, P, Q]
        wOut = tilerModel.getTensorDimVar(weightName, 0)  # C_out
        wIn  = tilerModel.getTensorDimVar(weightName, 1)  # C_in_per_group
        P    = tilerModel.getTensorDimVar(weightName, 2)  # kernel_h
        Q    = tilerModel.getTensorDimVar(weightName, 3)  # kernel_w

        # batch equal
        tilerModel.addConstraint(N_go == N_gi)

        # channel relations
        tilerModel.addConstraint(Co_go == wOut)
        tilerModel.addConstraint(Ci_gi == wIn * group)

        # spatial relation (standard conv output shape)
        # H_out = floor((H_in + pad_top + pad_bottom - P)/stride_h) + 1
        # W_out = floor((W_in + pad_left + pad_right - Q)/stride_w) + 1
        tilerModel.addConstraint(Ho_go == (Hi_gi + pad_top + pad_bottom - P) // stride_h + 1)
        tilerModel.addConstraint(Wo_go == (Wi_gi + pad_left + pad_right - Q) // stride_w + 1)

        return tilerModel

    # -----------------------
    # 2) Policy constraints
    # -----------------------
    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        gradOutBuf = ctxt.lookup(name=parseDict["data_in"])   # dY
        gradInBuf  = ctxt.lookup(name=parseDict["data_out"])  # dX
        weightBuf  = ctxt.lookup(name=parseDict["weight"])

        group = parseDict["group"]

        # NCHW gradIn vars (dX)
        C_gi = tilerModel.getTensorDimVar(gradInBuf.name, 1)
        H_gi = tilerModel.getTensorDimVar(gradInBuf.name, 2)
        W_gi = tilerModel.getTensorDimVar(gradInBuf.name, 3)

        # NCHW gradOut vars (dY)
        C_go = tilerModel.getTensorDimVar(gradOutBuf.name, 1)

        # Weight vars: [C_out, C_in_per_group, P, Q]
        wOut = tilerModel.getTensorDimVar(weightBuf.name, 0)
        wIn  = tilerModel.getTensorDimVar(weightBuf.name, 1)
        P    = tilerModel.getTensorDimVar(weightBuf.name, 2)
        Q    = tilerModel.getTensorDimVar(weightBuf.name, 3)

        # --- Must keep full channels because kernel does memset(dX) inside ---
        tilerModel.addConstraint(C_go == parseDict["ch_im_out"])   # full C_out
        tilerModel.addConstraint(C_gi == parseDict["ch_im_in"])    # full C_in

        # --- Kernel must be full ---
        # Your NodeTemplate passes (dim_kernel_x, dim_kernel_y) into (P, Q)
        tilerModel.addConstraint(P == parseDict["dim_kernel_x"])
        tilerModel.addConstraint(Q == parseDict["dim_kernel_y"])

        # --- Weight channel relations (full) ---
        tilerModel.addConstraint(wOut == parseDict["ch_im_out"])
        tilerModel.addConstraint(wIn * group == parseDict["ch_im_in"])

        # --- Minimum spatial tile sizes ---
        tilerModel.addConstraint(H_gi >= 1)
        tilerModel.addConstraint(W_gi >= 1)

        return tilerModel

    # -----------------------------------
    # 3) Symbolic node representation
    # -----------------------------------
    @staticmethod
    def constructSymbolicNodeRep(
        tilerModel: TilerModel,
        parseDict: Dict,
        ctxt: NetworkContext
    ) -> Dict[str, Union[int, IntVar]]:

        gradOutBuf = ctxt.lookup(name=parseDict["data_in"])    # dY
        weightBuf  = ctxt.lookup(name=parseDict["weight"])
        gradInBuf  = ctxt.lookup(name=parseDict["data_out"])   # dX

        symbolicParseDict = parseDict.copy()

        # gradOut (dY) NCHW: H/W/C at dimIdx 2/3/1
        symbolicParseDict["dim_im_out_x"] = tilerModel.getTensorDimVar(gradOutBuf.name, 2)  # H_out
        symbolicParseDict["dim_im_out_y"] = tilerModel.getTensorDimVar(gradOutBuf.name, 3)  # W_out
        symbolicParseDict["ch_im_out"]    = tilerModel.getTensorDimVar(gradOutBuf.name, 1)  # C_out

        # gradIn (dX) NCHW: H/W/C at dimIdx 2/3/1
        symbolicParseDict["dim_im_in_x"] = tilerModel.getTensorDimVar(gradInBuf.name, 2)   # H_in
        symbolicParseDict["dim_im_in_y"] = tilerModel.getTensorDimVar(gradInBuf.name, 3)   # W_in
        symbolicParseDict["ch_im_in"]    = tilerModel.getTensorDimVar(gradInBuf.name, 1)   # C_in

        # weight: [C_out, C_in_per_group, P, Q]
        symbolicParseDict["dim_kernel_x"] = tilerModel.getTensorDimVar(weightBuf.name, 2)  # P
        symbolicParseDict["dim_kernel_y"] = tilerModel.getTensorDimVar(weightBuf.name, 3)  # Q

        return symbolicParseDict

    # ---------------------------------------------------------
    # 4) Helper: gradIn(dX) tile -> required gradOut(dY) tile
    # ---------------------------------------------------------
    @staticmethod
    def computeGradOutCubeFromGradInTile(
        kernelShape: Tuple[int, int],              # (P, Q)
        pads: Tuple[int, int, int, int],           # (pad_top, pad_bottom, pad_left, pad_right)
        strides: Tuple[int, int],                  # (stride_h, stride_w)
        gradOutCSize: int,                         # full C_out
        gradInTile: HyperRectangle,                # tile on dX (output cube)
        gradInDims: Tuple[int, int, int, int],     # full dX dims (N, C_in, H_in, W_in)
        gradOutDims: Tuple[int, int, int, int],    # full dY dims (N, C_out, H_out, W_out)
    ) -> Tuple[HyperRectangle, Tuple[int, int, int, int]]:

        (nOff, _cOff_gi, hOff_gi, wOff_gi) = gradInTile.offset
        (nSize, _cSize_gi, hSize_gi, wSize_gi) = gradInTile.dims

        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides
        P, Q = kernelShape

        # For each ih, affected oh range: [ih*stride - pad_top, ih*stride - pad_top + P)
        oh0 = hOff_gi * stride_h - pad_top
        ow0 = wOff_gi * stride_w - pad_left
        oh1 = (hOff_gi + hSize_gi - 1) * stride_h - pad_top + P
        ow1 = (wOff_gi + wSize_gi - 1) * stride_w - pad_left + Q

        # clamp to gradOut valid range
        oh0_c = max(0, oh0)
        ow0_c = max(0, ow0)
        oh1_c = min(gradOutDims[2], oh1)
        ow1_c = min(gradOutDims[3], ow1)

        hSize_go = max(1, oh1_c - oh0_c)
        wSize_go = max(1, ow1_c - ow0_c)

        gradOutTile = HyperRectangle(
            (nOff, 0, oh0_c, ow0_c),                # C_out not tiled
            (nSize, gradOutCSize, hSize_go, wSize_go)
        )

        # Tile-level padding depends on whether gradIn tile touches gradIn boundary
        tile_pad_top    = pad_top    if hOff_gi == 0 else 0
        tile_pad_bottom = pad_bottom if (hOff_gi + hSize_gi) == gradInDims[2] else 0
        tile_pad_left   = pad_left   if wOff_gi == 0 else 0
        tile_pad_right  = pad_right  if (wOff_gi + wSize_gi) == gradInDims[3] else 0

        return gradOutTile, (tile_pad_top, tile_pad_bottom, tile_pad_left, tile_pad_right)

    # ---------------------------------------------------
    # 5) Serialize tiling solution into schedules + params
    # ---------------------------------------------------
    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        # Output cubes correspond to node output tensor => data_out => gradIn(dX)
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        varGradOut = operatorRepresentation["data_in"]   # dY
        varWeight  = operatorRepresentation["weight"]    # W
        varGradIn  = operatorRepresentation["data_out"]  # dX

        group   = operatorRepresentation["group"]
        pads    = operatorRepresentation["pads"]         # [t,b,l,r]
        strides = operatorRepresentation["strides"]      # [sh, sw]

        addrNames = ["data_in", "weight", "data_out"]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames
        )

        inputGradOutCubes: List[HyperRectangle] = []
        inputWeightCubes: List[HyperRectangle] = []

        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_in": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": [],
        }

        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t),
        }

        # Weight shape: [C_out, C_in_per_group, P, Q]
        (weightOutCh, weightInCh, weightP, weightQ) = ctxt.lookup(varWeight).shape

        fullGradInDims  = ctxt.lookup(varGradIn).shape   # (N, C_in, H_in, W_in)
        fullGradOutDims = ctxt.lookup(varGradOut).shape  # (N, C_out, H_out, W_out)

        stride_h, stride_w = strides

        for cube in outputCubes:
            # cube is gradIn(dX) tile: [N, C_in, H_in_tile, W_in_tile]
            (_nOff, _cOff, _hOff, _wOff) = cube.offset
            (_nSize, C_in_tile, H_in_tile, W_in_tile) = cube.dims

            # Needed gradOut(dY) tile for this gradIn tile
            gradOutCube, pad_tuple = cls.computeGradOutCubeFromGradInTile(
                kernelShape=(weightP, weightQ),
                pads=tuple(pads),
                strides=(stride_h, stride_w),
                gradOutCSize=weightOutCh,         # full C_out
                gradInTile=cube,
                gradInDims=fullGradInDims,
                gradOutDims=fullGradOutDims,
            )
            pad_top, pad_bottom, pad_left, pad_right = pad_tuple

            # Replacements for this tile
            replacements["dim_im_in_x"].append(H_in_tile)
            replacements["dim_im_in_y"].append(W_in_tile)

            replacements["dim_im_out_x"].append(gradOutCube.dims[2])  # H_out_tile
            replacements["dim_im_out_y"].append(gradOutCube.dims[3])  # W_out_tile

            # channels (should be full by policy constraints)
            replacements["ch_im_in"].append(C_in_tile)                # C_in
            replacements["ch_im_out"].append(weightOutCh)             # C_out

            replacements["padding_y_top"].append(pad_top)
            replacements["padding_y_bottom"].append(pad_bottom)
            replacements["padding_x_left"].append(pad_left)
            replacements["padding_x_right"].append(pad_right)

            inputGradOutCubes.append(gradOutCube)

            # Weight cube: full (since we don't tile C_out / C_in for this kernel)
            # layout: [C_out, C_in_per_group, P, Q]
            WeightCube = HyperRectangle((0, 0, 0, 0), (weightOutCh, weightInCh, weightP, weightQ))
            inputWeightCubes.append(WeightCube)

        # Build load schedules
        inputLoadSchedule = []
        outputLoadSchedule = []

        for go_cube, w_cube in zip(inputGradOutCubes, inputWeightCubes):
            inputLoadSchedule.append({"data_in": go_cube, "weight": w_cube})

        for out_cube in outputCubes:
            outputLoadSchedule.append({"data_out": out_cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule

class ConvGradW2DTileConstraint(TileConstraint):
    """
    ConvGradW (dW) TileConstraint for computing weight gradients (NCHW layout).

    Tensor mapping:
      input     = X (forward input)    [N, C_in,  H_in,  W_in]  NCHW
      data_in   = dY (grad_out)        [N, C_out, H_out, W_out] NCHW
      data_out  = dW (grad_weight)     [C_out, C_in_per_group, P, Q]

    Computation:
      dW[co, ci, kh, kw] = sum over (n, h_out, w_out) of:
          X[n, ci, h_out*stride + kh - pad_top, w_out*stride + kw - pad_left]
          * dY[n, co, h_out, w_out]

    Kernel behavior:
      - memset(dW, 0) before accumulation
      - Can tile over batch N (accumulate)
      - Can tile over spatial H_out/W_out (accumulate)
      - Weight output (dW) must be full (no tiling on C_out, C_in, P, Q)
    """

    # ---------------------------
    # 1) Geometrical constraints
    # ---------------------------
    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputName   = parseDict["data_in"]    # X (forward input)
        gradOutName = parseDict["grad_out"]   # dY
        gradWName   = parseDict["weight"]     # dW (output)

        tilerModel.addTensorDimToModel(ctxt, inputName)
        tilerModel.addTensorDimToModel(ctxt, gradOutName)
        tilerModel.addTensorDimToModel(ctxt, gradWName)

        pads    = parseDict["pads"]       # [pad_top, pad_bottom, pad_left, pad_right]
        strides = parseDict["strides"]    # [stride_h, stride_w]
        group   = parseDict["group"]

        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides

        # NCHW dims for input (X)
        N_x  = tilerModel.getTensorDimVar(inputName, 0)
        Ci_x = tilerModel.getTensorDimVar(inputName, 1)
        Hi_x = tilerModel.getTensorDimVar(inputName, 2)
        Wi_x = tilerModel.getTensorDimVar(inputName, 3)

        # NCHW dims for grad_out (dY)
        N_dy  = tilerModel.getTensorDimVar(gradOutName, 0)
        Co_dy = tilerModel.getTensorDimVar(gradOutName, 1)
        Ho_dy = tilerModel.getTensorDimVar(gradOutName, 2)
        Wo_dy = tilerModel.getTensorDimVar(gradOutName, 3)

        # grad_weight dims: [C_out, C_in_per_group, P, Q]
        Co_dw = tilerModel.getTensorDimVar(gradWName, 0)  # C_out
        Ci_dw = tilerModel.getTensorDimVar(gradWName, 1)  # C_in_per_group
        P     = tilerModel.getTensorDimVar(gradWName, 2)  # kernel_h
        Q     = tilerModel.getTensorDimVar(gradWName, 3)  # kernel_w

        # batch must match
        tilerModel.addConstraint(N_x == N_dy)

        # channel relations
        tilerModel.addConstraint(Ci_x == Ci_dw * group)   # input channels
        tilerModel.addConstraint(Co_dy == Co_dw)           # output channels

        # spatial relation (standard conv output shape)
        # H_out = floor((H_in + pad_top + pad_bottom - P) / stride_h) + 1
        # W_out = floor((W_in + pad_left + pad_right - Q) / stride_w) + 1
        tilerModel.addConstraint(Ho_dy == (Hi_x + pad_top + pad_bottom - P) // stride_h + 1)
        tilerModel.addConstraint(Wo_dy == (Wi_x + pad_left + pad_right - Q) // stride_w + 1)

        return tilerModel

    # -----------------------
    # 2) Policy constraints
    # -----------------------
    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuf    = ctxt.lookup(name=parseDict["data_in"])    # X
        gradOutBuf  = ctxt.lookup(name=parseDict["grad_out"])   # dY
        gradWBuf    = ctxt.lookup(name=parseDict["weight"])     # dW (output)

        group = parseDict["group"]

        # Input (X) vars - NCHW
        Ci_x = tilerModel.getTensorDimVar(inputBuf.name, 1)
        Hi_x = tilerModel.getTensorDimVar(inputBuf.name, 2)
        Wi_x = tilerModel.getTensorDimVar(inputBuf.name, 3)

        # GradOut (dY) vars - NCHW
        Co_dy = tilerModel.getTensorDimVar(gradOutBuf.name, 1)
        Ho_dy = tilerModel.getTensorDimVar(gradOutBuf.name, 2)
        Wo_dy = tilerModel.getTensorDimVar(gradOutBuf.name, 3)

        # GradWeight (dW) vars: [C_out, C_in_per_group, P, Q]
        Co_dw = tilerModel.getTensorDimVar(gradWBuf.name, 0)
        Ci_dw = tilerModel.getTensorDimVar(gradWBuf.name, 1)
        P     = tilerModel.getTensorDimVar(gradWBuf.name, 2)
        Q     = tilerModel.getTensorDimVar(gradWBuf.name, 3)

        # --- Weight output must be full (no tiling, needs accumulation) ---
        tilerModel.addConstraint(Co_dw == parseDict["ch_im_out"])
        tilerModel.addConstraint(Ci_dw * group == parseDict["ch_im_in"])
        tilerModel.addConstraint(P == parseDict["dim_kernel_x"])
        tilerModel.addConstraint(Q == parseDict["dim_kernel_y"])

        # --- Input channels must match ---
        tilerModel.addConstraint(Ci_x == parseDict["ch_im_in"])
        tilerModel.addConstraint(Co_dy == parseDict["ch_im_out"])

        # --- Minimum spatial tile sizes ---
        tilerModel.addConstraint(Hi_x >= 1)
        tilerModel.addConstraint(Wi_x >= 1)
        tilerModel.addConstraint(Ho_dy >= 1)
        tilerModel.addConstraint(Wo_dy >= 1)

        return tilerModel

    # -----------------------------------
    # 3) Symbolic node representation
    # -----------------------------------
    @staticmethod
    def constructSymbolicNodeRep(
        tilerModel: TilerModel,
        parseDict: Dict,
        ctxt: NetworkContext
    ) -> Dict[str, Union[int, IntVar]]:

        inputBuf   = ctxt.lookup(name=parseDict["data_in"])    # X
        gradOutBuf = ctxt.lookup(name=parseDict["grad_out"])   # dY
        gradWBuf   = ctxt.lookup(name=parseDict["weight"])     # dW (output)

        symbolicParseDict = parseDict.copy()

        # Input (X) NCHW: H/W/C at dimIdx 2/3/1
        symbolicParseDict["dim_im_in_x"] = tilerModel.getTensorDimVar(inputBuf.name, 2)   # H_in
        symbolicParseDict["dim_im_in_y"] = tilerModel.getTensorDimVar(inputBuf.name, 3)   # W_in
        symbolicParseDict["ch_im_in"]    = tilerModel.getTensorDimVar(inputBuf.name, 1)   # C_in

        # GradOut (dY) NCHW: H/W/C at dimIdx 2/3/1
        symbolicParseDict["dim_im_out_x"] = tilerModel.getTensorDimVar(gradOutBuf.name, 2)  # H_out
        symbolicParseDict["dim_im_out_y"] = tilerModel.getTensorDimVar(gradOutBuf.name, 3)  # W_out
        symbolicParseDict["ch_im_out"]    = tilerModel.getTensorDimVar(gradOutBuf.name, 1)  # C_out

        # GradWeight: [C_out, C_in_per_group, P, Q]
        symbolicParseDict["dim_kernel_x"] = tilerModel.getTensorDimVar(gradWBuf.name, 2)  # P
        symbolicParseDict["dim_kernel_y"] = tilerModel.getTensorDimVar(gradWBuf.name, 3)  # Q

        return symbolicParseDict

    # ---------------------------------------------------------
    # 4) Helper: compute required input tiles from grad_out tile
    # ---------------------------------------------------------
    @staticmethod
    def computeInputTileFromGradOutTile(
        kernelShape: Tuple[int, int],              # (P, Q)
        pads: Tuple[int, int, int, int],           # (pad_top, pad_bottom, pad_left, pad_right)
        strides: Tuple[int, int],                  # (stride_h, stride_w)
        inputCSize: int,                           # C_in
        gradOutTile: HyperRectangle,               # tile on dY
        inputDims: Tuple[int, int, int, int],      # full X dims (N, C_in, H_in, W_in)
        gradOutDims: Tuple[int, int, int, int],    # full dY dims (N, C_out, H_out, W_out)
    ) -> Tuple[HyperRectangle, Tuple[int, int, int, int]]:
        """
        Given a grad_out (dY) tile, compute the required input (X) tile.

        For each output position (h_out, w_out), we need input positions:
        [h_out*stride - pad_top, h_out*stride - pad_top + P)
        [w_out*stride - pad_left, w_out*stride - pad_left + Q)
        """
        (nOff, _cOff_dy, hOff_dy, wOff_dy) = gradOutTile.offset
        (nSize, _cSize_dy, hSize_dy, wSize_dy) = gradOutTile.dims

        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides
        P, Q = kernelShape

        # Compute required input range
        # For h_out in [hOff_dy, hOff_dy + hSize_dy),
        # need h_in in [h_out*stride - pad_top, h_out*stride - pad_top + P)

        h_in_start = hOff_dy * stride_h - pad_top
        w_in_start = wOff_dy * stride_w - pad_left

        h_in_end = (hOff_dy + hSize_dy - 1) * stride_h - pad_top + P
        w_in_end = (wOff_dy + wSize_dy - 1) * stride_w - pad_left + Q

        # Clamp to valid input range
        h_in_start_c = max(0, h_in_start)
        w_in_start_c = max(0, w_in_start)
        h_in_end_c = min(inputDims[2], h_in_end)
        w_in_end_c = min(inputDims[3], w_in_end)

        hSize_x = max(1, h_in_end_c - h_in_start_c)
        wSize_x = max(1, w_in_end_c - w_in_start_c)

        inputTile = HyperRectangle(
            (nOff, 0, h_in_start_c, w_in_start_c),  # C_in at full
            (nSize, inputCSize, hSize_x, wSize_x)
        )

        # Tile-level padding
        tile_pad_top    = pad_top    if hOff_dy == 0 else 0
        tile_pad_bottom = pad_bottom if (hOff_dy + hSize_dy) == gradOutDims[2] else 0
        tile_pad_left   = pad_left   if wOff_dy == 0 else 0
        tile_pad_right  = pad_right  if (wOff_dy + wSize_dy) == gradOutDims[3] else 0

        return inputTile, (tile_pad_top, tile_pad_bottom, tile_pad_left, tile_pad_right)

    # ---------------------------------------------------
    # 5) Serialize tiling solution into schedules + params
    # ---------------------------------------------------
    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        """
        For ConvGradW, output cubes correspond to grad_weight (dW).
        Since we don't tile the weight, there should be only one output cube (full weight).

        We tile the input (X) and grad_out (dY) over batch and/or spatial dimensions.
        """

        # Output cubes correspond to node output tensor => weight => grad_weight (dW)
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        varInput   = operatorRepresentation["data_in"]    # X
        varGradOut = operatorRepresentation["grad_out"]   # dY
        varGradW   = operatorRepresentation["weight"]     # dW (output)

        group   = operatorRepresentation["group"]
        pads    = operatorRepresentation["pads"]         # [t,b,l,r]
        strides = operatorRepresentation["strides"]      # [sh, sw]

        addrNames = ["data_in", "grad_out", "weight"]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames
        )

        inputXCubes: List[HyperRectangle] = []
        inputGradOutCubes: List[HyperRectangle] = []

        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_in": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": [],
        }

        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t),
        }

        # Weight shape: [C_out, C_in_per_group, P, Q]
        (weightCo, weightCi, weightP, weightQ) = ctxt.lookup(varGradW).shape

        fullInputDims   = ctxt.lookup(varInput).shape     # (N, C_in, H_in, W_in)
        fullGradOutDims = ctxt.lookup(varGradOut).shape   # (N, C_out, H_out, W_out)

        stride_h, stride_w = strides

        # Note: For ConvGradW, we typically don't tile the weight output.
        # Instead, we tile over batch/spatial dimensions of input and grad_out.
        # The output cube here represents the full weight gradient.

        # If policy constraints are correct, there should be only one output cube (full dW)
        assert len(outputCubes) == 1, "ConvGradW should have only one output cube (full weight)"

        # Since we're tiling input/grad_out but not weight, we need to define
        # how to break up the computation. Typically, we tile grad_out and compute
        # corresponding input tiles.

        # For simplicity, assume grad_out is tiled and we compute required input tiles
        # This is a simplified approach - actual tiling may need custom logic

        # Get the single weight output cube (should be full weight)
        dw_cube = outputCubes[0]

        # For now, create a single tile covering full grad_out and input
        # In practice, you'd tile grad_out over batch/spatial dimensions

        # Full grad_out cube
        gradOutCube = HyperRectangle(
            (0, 0, 0, 0),
            fullGradOutDims
        )

        # Corresponding input cube
        inputCube, pad_tuple = cls.computeInputTileFromGradOutTile(
            kernelShape=(weightP, weightQ),
            pads=tuple(pads),
            strides=(stride_h, stride_w),
            inputCSize=fullInputDims[1],  # C_in
            gradOutTile=gradOutCube,
            inputDims=fullInputDims,
            gradOutDims=fullGradOutDims,
        )

        pad_top, pad_bottom, pad_left, pad_right = pad_tuple

        # Replacements for this tile
        replacements["dim_im_in_x"].append(inputCube.dims[2])    # H_in
        replacements["dim_im_in_y"].append(inputCube.dims[3])    # W_in
        replacements["dim_im_out_x"].append(gradOutCube.dims[2]) # H_out
        replacements["dim_im_out_y"].append(gradOutCube.dims[3]) # W_out
        replacements["ch_im_in"].append(fullInputDims[1])        # C_in
        replacements["ch_im_out"].append(fullGradOutDims[1])     # C_out

        replacements["padding_y_top"].append(pad_top)
        replacements["padding_y_bottom"].append(pad_bottom)
        replacements["padding_x_left"].append(pad_left)
        replacements["padding_x_right"].append(pad_right)

        inputXCubes.append(inputCube)
        inputGradOutCubes.append(gradOutCube)

        # Build load schedules
        inputLoadSchedule = []
        outputLoadSchedule = []

        for x_cube, dy_cube in zip(inputXCubes, inputGradOutCubes):
            inputLoadSchedule.append({"data_in": x_cube, "grad_out": dy_cube})

        for out_cube in outputCubes:
            outputLoadSchedule.append({"weight": out_cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule

class DWConvGradX2DTileConstraint(TileConstraint):
    """
    DWConvGradX (depthwise) uses REVERSED dimension mapping compared to ConvGradX!

    PULPDWConvGradX2DParser maps dimensions opposite to regular ConvGradX:
      - ch_im_in/dim_im_in_x/y refer to grad_out (C_out, H_out, W_out)
      - ch_im_out/dim_im_out_x/y refer to grad_in (C_in, H_in, W_in)

    Tensor mapping:
      data_in  = grad_out [N, C_out, H_out, W_out] (smaller)
      data_out = grad_in  [N, C_in,  H_in,  W_in]  (larger)
      weight   = W [C_in, 1, P, Q] for depthwise
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        gradOutName = parseDict["data_in"]    # grad_out (smaller)
        gradInName  = parseDict["data_out"]   # grad_in (larger)
        weightName  = parseDict["weight"]

        tilerModel.addTensorDimToModel(ctxt, gradOutName)
        tilerModel.addTensorDimToModel(ctxt, gradInName)
        tilerModel.addTensorDimToModel(ctxt, weightName)

        pads    = parseDict["pads"]
        strides = parseDict["strides"]
        group   = parseDict["group"]
        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides

        # grad_out (data_in, smaller)
        N_go  = tilerModel.getTensorDimVar(gradOutName, 0)
        Co_go = tilerModel.getTensorDimVar(gradOutName, 1)
        Ho_go = tilerModel.getTensorDimVar(gradOutName, 2)
        Wo_go = tilerModel.getTensorDimVar(gradOutName, 3)

        # grad_in (data_out, larger)
        N_gi  = tilerModel.getTensorDimVar(gradInName, 0)
        Ci_gi = tilerModel.getTensorDimVar(gradInName, 1)
        Hi_gi = tilerModel.getTensorDimVar(gradInName, 2)
        Wi_gi = tilerModel.getTensorDimVar(gradInName, 3)

        # Weight: [C_in, 1, P, Q] for DW
        wCin = tilerModel.getTensorDimVar(weightName, 0)
        wCpg = tilerModel.getTensorDimVar(weightName, 1)
        P    = tilerModel.getTensorDimVar(weightName, 2)
        Q    = tilerModel.getTensorDimVar(weightName, 3)

        # Constraints
        tilerModel.addConstraint(N_go == N_gi)
        tilerModel.addConstraint(Co_go == wCin)
        tilerModel.addConstraint(Ci_gi == wCin)
        tilerModel.addConstraint(wCpg == 1)
        tilerModel.addConstraint(Ho_go == (Hi_gi + pad_top + pad_bottom - P) // stride_h + 1)
        tilerModel.addConstraint(Wo_go == (Wi_gi + pad_left + pad_right - Q) // stride_w + 1)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        gradOutBuf = ctxt.lookup(name=parseDict["data_in"])
        gradInBuf  = ctxt.lookup(name=parseDict["data_out"])
        weightBuf  = ctxt.lookup(name=parseDict["weight"])

        Co_go = tilerModel.getTensorDimVar(gradOutBuf.name, 1)
        Ci_gi = tilerModel.getTensorDimVar(gradInBuf.name, 1)
        Hi_gi = tilerModel.getTensorDimVar(gradInBuf.name, 2)
        Wi_gi = tilerModel.getTensorDimVar(gradInBuf.name, 3)

        P = tilerModel.getTensorDimVar(weightBuf.name, 2)
        Q = tilerModel.getTensorDimVar(weightBuf.name, 3)

        # Note: reversed mapping in parseDict!
        tilerModel.addConstraint(Co_go == parseDict["ch_im_in"])    # grad_out
        tilerModel.addConstraint(Ci_gi == parseDict["ch_im_out"])   # grad_in
        tilerModel.addConstraint(P == parseDict["dim_kernel_x"])
        tilerModel.addConstraint(Q == parseDict["dim_kernel_y"])
        tilerModel.addConstraint(Hi_gi >= 1)
        tilerModel.addConstraint(Wi_gi >= 1)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        # Parser already set up dimensions correctly with reversed mapping
        return parseDict.copy()

    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        """
        DWConvGradX serialization - note reversed dimension mapping!
        Output cubes = grad_in (data_out, larger)
        Input cubes = grad_out (data_in, smaller)
        """

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        varGradOut = operatorRepresentation["data_in"]    # grad_out (smaller)
        varWeight  = operatorRepresentation["weight"]
        varGradIn  = operatorRepresentation["data_out"]   # grad_in (larger)

        group   = operatorRepresentation["group"]
        pads    = operatorRepresentation["pads"]
        strides = operatorRepresentation["strides"]

        addrNames = ["data_in", "weight", "data_out"]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames
        )

        inputGradOutCubes: List[HyperRectangle] = []
        inputWeightCubes: List[HyperRectangle] = []

        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_in": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": [],
        }

        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t),
        }

        # Weight shape: [C_in, 1, P, Q] for DW
        (weightCin, weightCpg, weightP, weightQ) = ctxt.lookup(varWeight).shape

        fullGradInDims  = ctxt.lookup(varGradIn).shape    # (N, C_in, H_in, W_in) - larger
        fullGradOutDims = ctxt.lookup(varGradOut).shape   # (N, C_out, H_out, W_out) - smaller

        stride_h, stride_w = strides

        for cube in outputCubes:
            # cube is grad_in tile: [N, C_in, H_in_tile, W_in_tile] (larger)
            (_nOff, _cOff, hOff_gi, wOff_gi) = cube.offset
            (_nSize, C_in_tile, H_in_tile, W_in_tile) = cube.dims

            # Compute needed grad_out tile (smaller) from grad_in tile (larger)
            # Use same logic as ConvGradX's computeGradOutCubeFromGradInTile
            pad_top, pad_bottom, pad_left, pad_right = pads

            oh0 = hOff_gi * stride_h - pad_top
            ow0 = wOff_gi * stride_w - pad_left
            oh1 = (hOff_gi + H_in_tile - 1) * stride_h - pad_top + weightP
            ow1 = (wOff_gi + W_in_tile - 1) * stride_w - pad_left + weightQ

            oh0_c = max(0, oh0)
            ow0_c = max(0, ow0)
            oh1_c = min(fullGradOutDims[2], oh1)
            ow1_c = min(fullGradOutDims[3], ow1)

            hSize_go = max(1, oh1_c - oh0_c)
            wSize_go = max(1, ow1_c - ow0_c)

            gradOutCube = HyperRectangle(
                (_nOff, 0, oh0_c, ow0_c),
                (_nSize, weightCin, hSize_go, wSize_go)  # C_out = C_in for DW
            )

            # Tile-level padding
            tile_pad_top    = pad_top    if hOff_gi == 0 else 0
            tile_pad_bottom = pad_bottom if (hOff_gi + H_in_tile) == fullGradInDims[2] else 0
            tile_pad_left   = pad_left   if wOff_gi == 0 else 0
            tile_pad_right  = pad_right  if (wOff_gi + W_in_tile) == fullGradInDims[3] else 0

            # CRITICAL: DWConvGradX has REVERSED mapping in parseDict!
            # ch_im_in = C_out (grad_out), ch_im_out = C_in (grad_in)
            # dim_im_in = H_out/W_out, dim_im_out = H_in/W_in
            replacements["dim_im_in_x"].append(hSize_go)      # grad_out H (reversed!)
            replacements["dim_im_in_y"].append(wSize_go)      # grad_out W
            replacements["dim_im_out_x"].append(H_in_tile)    # grad_in H
            replacements["dim_im_out_y"].append(W_in_tile)    # grad_in W
            replacements["ch_im_in"].append(weightCin)        # C_out (reversed!)
            replacements["ch_im_out"].append(C_in_tile)       # C_in

            replacements["padding_y_top"].append(tile_pad_top)
            replacements["padding_y_bottom"].append(tile_pad_bottom)
            replacements["padding_x_left"].append(tile_pad_left)
            replacements["padding_x_right"].append(tile_pad_right)

            inputGradOutCubes.append(gradOutCube)

            # Weight cube: full [C_in, 1, P, Q]
            WeightCube = HyperRectangle((0, 0, 0, 0), (weightCin, 1, weightP, weightQ))
            inputWeightCubes.append(WeightCube)

        # Build load schedules
        inputLoadSchedule = []
        outputLoadSchedule = []

        for go_cube, w_cube in zip(inputGradOutCubes, inputWeightCubes):
            inputLoadSchedule.append({"data_in": go_cube, "weight": w_cube})

        for out_cube in outputCubes:
            outputLoadSchedule.append({"data_out": out_cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule

class DWConvGradW2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputName   = parseDict["data_in"]
        gradOutName = parseDict["grad_out"]
        gradWName   = parseDict["weight"]

        tilerModel.addTensorDimToModel(ctxt, inputName)
        tilerModel.addTensorDimToModel(ctxt, gradOutName)
        tilerModel.addTensorDimToModel(ctxt, gradWName)

        pads    = parseDict["pads"]
        strides = parseDict["strides"]
        group   = parseDict["group"]

        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides

        N_x  = tilerModel.getTensorDimVar(inputName, 0)
        Ci_x = tilerModel.getTensorDimVar(inputName, 1)
        Hi_x = tilerModel.getTensorDimVar(inputName, 2)
        Wi_x = tilerModel.getTensorDimVar(inputName, 3)

        N_dy  = tilerModel.getTensorDimVar(gradOutName, 0)
        Co_dy = tilerModel.getTensorDimVar(gradOutName, 1)
        Ho_dy = tilerModel.getTensorDimVar(gradOutName, 2)
        Wo_dy = tilerModel.getTensorDimVar(gradOutName, 3)

        Co_dw = tilerModel.getTensorDimVar(gradWName, 0)
        Ci_dw = tilerModel.getTensorDimVar(gradWName, 1)
        P     = tilerModel.getTensorDimVar(gradWName, 2)
        Q     = tilerModel.getTensorDimVar(gradWName, 3)

        tilerModel.addConstraint(N_x == N_dy)
        tilerModel.addConstraint(Ci_dw == 1)
        tilerModel.addConstraint(Ci_x == Co_dw)
        tilerModel.addConstraint(Co_dy == Co_dw)
        tilerModel.addConstraint(Ho_dy == (Hi_x + pad_top + pad_bottom - P) // stride_h + 1)
        tilerModel.addConstraint(Wo_dy == (Wi_x + pad_left + pad_right - Q) // stride_w + 1)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuf    = ctxt.lookup(name=parseDict["data_in"])
        gradOutBuf  = ctxt.lookup(name=parseDict["grad_out"])
        gradWBuf    = ctxt.lookup(name=parseDict["weight"])

        group = parseDict["group"]

        Ci_x = tilerModel.getTensorDimVar(inputBuf.name, 1)
        Hi_x = tilerModel.getTensorDimVar(inputBuf.name, 2)
        Wi_x = tilerModel.getTensorDimVar(inputBuf.name, 3)

        Co_dy = tilerModel.getTensorDimVar(gradOutBuf.name, 1)
        Ho_dy = tilerModel.getTensorDimVar(gradOutBuf.name, 2)
        Wo_dy = tilerModel.getTensorDimVar(gradOutBuf.name, 3)

        Co_dw = tilerModel.getTensorDimVar(gradWBuf.name, 0)
        Ci_dw = tilerModel.getTensorDimVar(gradWBuf.name, 1)
        P     = tilerModel.getTensorDimVar(gradWBuf.name, 2)
        Q     = tilerModel.getTensorDimVar(gradWBuf.name, 3)

        tilerModel.addConstraint(Co_dw == parseDict["ch_im_out"])
        tilerModel.addConstraint(Ci_dw == 1)
        tilerModel.addConstraint(P == parseDict["dim_kernel_x"])
        tilerModel.addConstraint(Q == parseDict["dim_kernel_y"])
        tilerModel.addConstraint(Ci_x == parseDict["ch_im_in"])
        tilerModel.addConstraint(Co_dy == parseDict["ch_im_out"])
        tilerModel.addConstraint(Ci_x == Co_dy)
        tilerModel.addConstraint(Hi_x >= 1)
        tilerModel.addConstraint(Wi_x >= 1)
        tilerModel.addConstraint(Ho_dy >= 1)
        tilerModel.addConstraint(Wo_dy >= 1)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(
        tilerModel: TilerModel,
        parseDict: Dict,
        ctxt: NetworkContext
    ) -> Dict[str, Union[int, IntVar]]:
        # DWConvGradW kernel expects (y, x) parameter order, different from regular ConvGradW
        # Parser already set up dimensions correctly, just return as-is
        return parseDict.copy()

    @staticmethod
    def computeInputTileFromGradOutTile(
        kernelShape: Tuple[int, int],
        pads: Tuple[int, int, int, int],
        strides: Tuple[int, int],
        inputCSize: int,
        gradOutTile: HyperRectangle,
        inputDims: Tuple[int, int, int, int],
        gradOutDims: Tuple[int, int, int, int],
    ) -> Tuple[HyperRectangle, Tuple[int, int, int, int]]:
        (nOff, _cOff_dy, hOff_dy, wOff_dy) = gradOutTile.offset
        (nSize, _cSize_dy, hSize_dy, wSize_dy) = gradOutTile.dims

        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides
        P, Q = kernelShape

        h_in_start = hOff_dy * stride_h - pad_top
        w_in_start = wOff_dy * stride_w - pad_left
        h_in_end = (hOff_dy + hSize_dy - 1) * stride_h - pad_top + P
        w_in_end = (wOff_dy + wSize_dy - 1) * stride_w - pad_left + Q

        h_in_start_c = max(0, h_in_start)
        w_in_start_c = max(0, w_in_start)
        h_in_end_c = min(inputDims[2], h_in_end)
        w_in_end_c = min(inputDims[3], w_in_end)

        hSize_x = max(1, h_in_end_c - h_in_start_c)
        wSize_x = max(1, w_in_end_c - w_in_start_c)

        inputTile = HyperRectangle(
            (nOff, 0, h_in_start_c, w_in_start_c),
            (nSize, inputCSize, hSize_x, wSize_x)
        )

        tile_pad_top    = pad_top    if hOff_dy == 0 else 0
        tile_pad_bottom = pad_bottom if (hOff_dy + hSize_dy) == gradOutDims[2] else 0
        tile_pad_left   = pad_left   if wOff_dy == 0 else 0
        tile_pad_right  = pad_right  if (wOff_dy + wSize_dy) == gradOutDims[3] else 0

        return inputTile, (tile_pad_top, tile_pad_bottom, tile_pad_left, tile_pad_right)

    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        varInput   = operatorRepresentation["data_in"]
        varGradOut = operatorRepresentation["grad_out"]
        varGradW   = operatorRepresentation["weight"]

        pads    = operatorRepresentation["pads"]
        strides = operatorRepresentation["strides"]

        # Base address extraction (standard)
        addrNames = ["data_in", "grad_out", "weight"]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames
        )

        # Full shapes
        fullInputDims   = ctxt.lookup(varInput).shape     # [N, C, H, W]
        fullGradOutDims = ctxt.lookup(varGradOut).shape   # [N, C, Ho, Wo]
        fullWeightDims  = ctxt.lookup(varGradW).shape     # [C, 1, P, Q]

        # Prepare replacements (scalar single entry)
        replacements = {
            "dim_im_in_x":  [fullInputDims[3]],   # W_in
            "dim_im_in_y":  [fullInputDims[2]],   # H_in
            "dim_im_out_x": [fullGradOutDims[3]], # W_out
            "dim_im_out_y": [fullGradOutDims[2]], # H_out
            "ch_im_in":     [fullInputDims[1]],
            "ch_im_out":    [fullGradOutDims[1]],
            "padding_y_top":    [pads[0]],
            "padding_y_bottom": [pads[2]],
            "padding_x_left":   [pads[1]],
            "padding_x_right":  [pads[3]],
        }

        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t),
        }

        # Full tile (identity mapping, no tiling)
        inputLoadSchedule = [{
            "grad_out": HyperRectangle((0,0,0,0), fullGradOutDims),
            "data_in": HyperRectangle((0,0,0,0), fullInputDims),
        }]

        outputLoadSchedule = [{
            "weight": HyperRectangle((0,0,0,0), fullWeightDims),
        }]

        tilingSchedule = TilingSchedule(
            inputBaseOffsets,
            outputBaseOffsets,
            inputLoadSchedule,
            outputLoadSchedule
        )

        variableReplacementSchedule = VariableReplacementScheme(
            replacements,
            replacementTypes
        )

        return variableReplacementSchedule, tilingSchedule