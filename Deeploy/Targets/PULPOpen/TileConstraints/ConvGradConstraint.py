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


class ConvGradXTileConstraintBase(TileConstraint):
    """
    Base for ConvGradX2D tiling:

      - absoluteOutputCubes are tiles of grad_in (dX)  (operatorRepresentation[gradInKey])
      - for each dX tile, derive required grad_out (dY) halo tile
      - weight is full (not tiled)
      - emits unified template params:
          ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
          ${offset_grad_in_h}, ${offset_grad_in_w}, ${offset_grad_out_h}, ${offset_grad_out_w}
    """

    # ---- parser/opRep keys (override in subclasses if needed) ----
    # In Deeploy ConvGradX parsers these are commonly "data_in" (dY) and "data_out" (dX).
    gradOutKey = "grad_out"   # dY
    gradInKey  = "grad_in"  # dX
    weightKey  = "weight"    # W

    # ---------------------------
    # 1) Geometrical constraints
    # ---------------------------
    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dyName = parseDict[cls.gradOutKey]
        dxName = parseDict[cls.gradInKey]
        wName  = parseDict[cls.weightKey]

        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dxName)
        tilerModel.addTensorDimToModel(ctxt, wName)

        group = parseDict.get("group", 1)

        # N match
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 0) == tilerModel.getTensorDimVar(dxName, 0))

        # Channel relations:
        # dY: [N, C_out, H_out, W_out]
        # dX: [N, C_in,  H_in,  W_in]
        # W : [C_out, C_in/group, P, Q]
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == tilerModel.getTensorDimVar(wName, 0))
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dxName, 1) == tilerModel.getTensorDimVar(wName, 1) * group)

        return tilerModel

    # -----------------------
    # 2) Policy constraints
    # -----------------------
    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Default policy:
          - keep full Cin/Cout
          - weight not tiled
          - allow spatial tiling on dX
        """
        dyName = parseDict[cls.gradOutKey]
        dxName = parseDict[cls.gradInKey]
        wName  = parseDict[cls.weightKey]

        dyBuf = ctxt.lookup(dyName)
        dxBuf = ctxt.lookup(dxName)
        wBuf  = ctxt.lookup(wName)

        # full channels
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == dyBuf.shape[1])  # Cout full
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dxName, 1) == dxBuf.shape[1])  # Cin full

        # weight not tiled
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 0) == wBuf.shape[0])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 1) == wBuf.shape[1])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 2) == wBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 3) == wBuf.shape[3])

        return tilerModel

    # -----------------------------------
    # 3) Symbolic node representation
    # -----------------------------------
    @classmethod
    def constructSymbolicNodeRep(
        cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext
    ) -> Dict[str, Union[int, IntVar]]:
        """
        Bind template fields:
          dim_im_out_* / ch_im_out : grad_out (dY)
          dim_im_in_*  / ch_im_in  : grad_in  (dX)
        """
        dyName = parseDict[cls.gradOutKey]
        dxName = parseDict[cls.gradInKey]
        wName  = parseDict[cls.weightKey]

        symbolic = parseDict.copy()

        # dY (grad_out)
        symbolic["dim_im_out_x"] = tilerModel.getTensorDimVar(dyName, 2)  # H_out tile
        symbolic["dim_im_out_y"] = tilerModel.getTensorDimVar(dyName, 3)  # W_out tile
        symbolic["ch_im_out"]    = tilerModel.getTensorDimVar(dyName, 1)  # Cout

        # dX (grad_in)
        symbolic["dim_im_in_x"] = tilerModel.getTensorDimVar(dxName, 2)   # H_in tile
        symbolic["dim_im_in_y"] = tilerModel.getTensorDimVar(dxName, 3)   # W_in tile
        symbolic["ch_im_in"]    = tilerModel.getTensorDimVar(dxName, 1)   # Cin

        # kernel (H,W)
        symbolic["dim_kernel_x"] = tilerModel.getTensorDimVar(wName, 2)   # P
        symbolic["dim_kernel_y"] = tilerModel.getTensorDimVar(wName, 3)   # Q

        # offsets filled in serialize
        symbolic["offset_grad_in_h"] = 0
        symbolic["offset_grad_in_w"] = 0
        symbolic["offset_grad_out_h"] = 0
        symbolic["offset_grad_out_w"] = 0

        return symbolic

    # -------------------------------
    # helpers
    # -------------------------------
    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return -((-a) // b)

    @staticmethod
    def _floor_div(a: int, b: int) -> int:
        return a // b

    @classmethod
    def get_kernel_hw(cls, ctxt: NetworkContext, wName: str, wShape: Tuple[int, int, int, int]) -> Tuple[int, int]:
        return wShape[2], wShape[3]

    @classmethod
    def get_dy_channels(
        cls,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
        dyName: str,
        dxName: str,
        wName: str,
        dyFull: Tuple[int, int, int, int],
        dxFull: Tuple[int, int, int, int],
        wShape: Tuple[int, int, int, int],
    ) -> int:
        # default ConvGradX: dy channels == weight[0] (Cout)
        return wShape[0]

    @classmethod
    def get_ch_im_out(cls, ctxt: NetworkContext, dyFull, dxFull, wShape) -> int:
        # template's ch_im_out should match dY channels (Cout)
        return dyFull[1]

    @classmethod
    def get_ch_im_in(cls, ctxt: NetworkContext, dyFull, dxFull, wShape) -> int:
        # template's ch_im_in should match dX channels (Cin)
        return dxFull[1]

    @classmethod
    def map_onnx_pads_to_template(cls, tpt: int, tpb: int, tpl: int, tpr: int) -> Tuple[int, int, int, int]:
        """
        ONNX pads are (top,bottom,left,right) where top/bottom are H, left/right are W.

        Template wants unified order:
          (${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right})
        with your convention:
          y -> W dimension, x -> H dimension

        So:
          padding_y_top    = left
          padding_y_bottom = right
          padding_x_left   = top
          padding_x_right  = bottom
        """
        return (tpl, tpr, tpt, tpb)

    @classmethod
    def computeDyCubeFromDxTile(
        cls,
        dxTile: HyperRectangle,                 # (N,Cin,Hx,Wx)
        dyFull: Tuple[int, int, int, int],      # full dY
        P: int,
        Q: int,
        pads: Tuple[int, int, int, int],        # (t,b,l,r)
        strides: Tuple[int, int],               # (sh, sw)
        dyC: int,                               # Cout for this op
        dxAbsOff: Tuple[int, int, int, int],    # abs offset for boundary decision
    ) -> Tuple[HyperRectangle, Tuple[int, int, int, int]]:

        (nOff, _cOff, _hxOff_rel, _wxOff_rel) = dxTile.offset
        (nSize, _cinSize, hxSize, wxSize) = dxTile.dims

        (_, _, hxOff_abs, wxOff_abs) = dxAbsOff

        pad_top, pad_bottom, pad_left, pad_right = pads
        sh, sw = strides

        hx0 = hxOff_abs
        hx1 = hxOff_abs + hxSize - 1
        wx0 = wxOff_abs
        wx1 = wxOff_abs + wxSize - 1

        Hy = dyFull[2]
        Wy = dyFull[3]

        oy0 = cls._ceil_div(hx0 - (P - 1) + pad_top, sh)
        oy1 = cls._floor_div(hx1 + pad_top, sh)
        ox0 = cls._ceil_div(wx0 - (Q - 1) + pad_left, sw)
        ox1 = cls._floor_div(wx1 + pad_left, sw)

        oy0 = max(0, oy0)
        ox0 = max(0, ox0)
        oy1 = min(Hy - 1, oy1)
        ox1 = min(Wy - 1, ox1)

        if oy0 > oy1 or ox0 > ox1:
            raise RuntimeError(
                f"dx tile {dxTile.offset}/{dxTile.dims} produces empty dy halo: "
                f"oy[{oy0},{oy1}] ox[{ox0},{ox1}] (Hy={Hy},Wy={Wy},P={P},Q={Q},pads={pads},strides={strides})"
            )

        dyH = oy1 - oy0 + 1
        dyW = ox1 - ox0 + 1

        dyCube = HyperRectangle(
            (nOff, 0, oy0, ox0),     # dY: (N, C_out, H, W)
            (nSize, dyC, dyH, dyW)
        )

        # tile-level ONNX pads only at boundary
        tile_pad_top    = pad_top    if oy0 == 0 else 0
        tile_pad_bottom = pad_bottom if (oy0 + dyH) == Hy else 0
        tile_pad_left   = pad_left   if ox0 == 0 else 0
        tile_pad_right  = pad_right  if (ox0 + dyW) == Wy else 0

        return dyCube, (tile_pad_top, tile_pad_bottom, tile_pad_left, tile_pad_right)

    @staticmethod
    def _get_abs_off(abs_obj: AbsoluteHyperRectangle, fallback_rect: HyperRectangle):
        abs_off = getattr(abs_obj, "absoluteOffset", None)
        if abs_off is None:
            abs_off = getattr(abs_obj, "absolute_offset", None)
        if abs_off is None:
            abs_off = fallback_rect.offset
        return abs_off

    @classmethod
    def extraSerializeChecks(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> None:
        """Hook for DW checks etc."""
        return

    # ---------------------------------------------------
    # 4) serialize: dx tiles -> dy halo tiles
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

        cls.extraSerializeChecks(ctxt, operatorRepresentation)

        varDY = operatorRepresentation[cls.gradOutKey]  # dY
        varW  = operatorRepresentation[cls.weightKey]   # W
        varDX = operatorRepresentation[cls.gradInKey]   # dX

        pads    = tuple(operatorRepresentation.get("pads", [0, 0, 0, 0]))   # (t,b,l,r)
        strides = tuple(operatorRepresentation.get("strides", [1, 1]))      # (sh,sw)

        dyFull = tuple(ctxt.lookup(varDY).shape)  # (N,Cout,Ho,Wo)
        dxFull = tuple(ctxt.lookup(varDX).shape)  # (N,Cin,Hi,Wi)
        wShape = tuple(ctxt.lookup(varW).shape)   # (Cout,Cin/group,P,Q) or DW: (Cin,1,P,Q)

        P, Q = cls.get_kernel_hw(ctxt, varW, wShape)
        dyC = cls.get_dy_channels(ctxt, operatorRepresentation, varDY, varDX, varW, dyFull, dxFull, wShape)

        dxTiles = [c.rectangle for c in absoluteOutputCubes]

        addrNames = [cls.gradOutKey, cls.weightKey, cls.gradInKey]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames
        )

        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_in": [],
            "ch_im_out": [],

            # unified template order:
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": [],

            "offset_grad_in_h": [],
            "offset_grad_in_w": [],
            "offset_grad_out_h": [],
            "offset_grad_out_w": [],
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

            "offset_grad_in_h": PointerClass(uint16_t),
            "offset_grad_in_w": PointerClass(uint16_t),
            "offset_grad_out_h": PointerClass(uint16_t),
            "offset_grad_out_w": PointerClass(uint16_t),
        }

        inputDyCubes: List[HyperRectangle] = []
        inputWCubes:  List[HyperRectangle] = []
        outputDxCubes: List[HyperRectangle] = []

        fullW = HyperRectangle((0, 0, 0, 0), wShape)

        ch_in  = cls.get_ch_im_in(ctxt, dyFull, dxFull, wShape)
        ch_out = cls.get_ch_im_out(ctxt, dyFull, dxFull, wShape)

        for idx, dxCube in enumerate(dxTiles):
            abs_off = cls._get_abs_off(absoluteOutputCubes[idx], dxCube)

            dyCube, (tpt, tpb, tpl, tpr) = cls.computeDyCubeFromDxTile(
                dxTile=dxCube,
                dyFull=dyFull,
                P=P, Q=Q,
                pads=pads,
                strides=strides,
                dyC=dyC,              # IMPORTANT: use computed dyC
                dxAbsOff=abs_off
            )

            replacements["dim_im_in_x"].append(dxCube.dims[2])    # H_in_tile
            replacements["dim_im_in_y"].append(dxCube.dims[3])    # W_in_tile
            replacements["dim_im_out_x"].append(dyCube.dims[2])   # H_out_tile (halo)
            replacements["dim_im_out_y"].append(dyCube.dims[3])   # W_out_tile (halo)

            replacements["ch_im_in"].append(ch_in)
            replacements["ch_im_out"].append(ch_out)

            py_top, py_bottom, px_left, px_right = cls.map_onnx_pads_to_template(tpt, tpb, tpl, tpr)
            replacements["padding_y_top"].append(py_top)
            replacements["padding_y_bottom"].append(py_bottom)
            replacements["padding_x_left"].append(px_left)
            replacements["padding_x_right"].append(px_right)

            replacements["offset_grad_in_h"].append(abs_off[2])
            replacements["offset_grad_in_w"].append(abs_off[3])
            replacements["offset_grad_out_h"].append(dyCube.offset[2])
            replacements["offset_grad_out_w"].append(dyCube.offset[3])

            inputDyCubes.append(dyCube)
            inputWCubes.append(fullW)
            outputDxCubes.append(dxCube)

        inputLoadSchedule  = [{cls.gradOutKey: dy, cls.weightKey: w} for dy, w in zip(inputDyCubes, inputWCubes)]
        outputLoadSchedule = [{cls.gradInKey: dx} for dx in outputDxCubes]

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)
        return variableReplacementSchedule, tilingSchedule


# ============================================================================
# ConvGradX: subclass reusing the base
# ============================================================================

class ConvGradX2DHWTileConstraint(ConvGradXTileConstraintBase):
    pass


class ConvGradX2DIm2ColHWTileConstraint(ConvGradXTileConstraintBase):
    pass

class PWConvGradXTileConstraint(ConvGradXTileConstraintBase):
    pass

class DWConvGradX2DTileConstraint(ConvGradXTileConstraintBase):
    """
    Depthwise ConvGradX (dX) tiling, reusing ConvGradXTileConstraintBase.

    Expected tensors:
      data_in  = grad_out (dY) [N, C, H_out, W_out]
      data_out = grad_in  (dX) [N, C, H_in,  W_in]
      weight   = W        [C, 1, P, Q]
    """

    # If your DW parser uses different keys, override here.
    gradOutKey = "grad_out"
    gradInKey  = "grad_in"
    weightKey  = "weight"

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        dyName = parseDict[cls.gradOutKey]  # dY
        dxName = parseDict[cls.gradInKey]   # dX
        wName  = parseDict[cls.weightKey]   # W

        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dxName)
        tilerModel.addTensorDimToModel(ctxt, wName)

        # N match
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 0) == tilerModel.getTensorDimVar(dxName, 0))

        # DW channels: Cin == Cout == W[0], and W[1]==1
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == tilerModel.getTensorDimVar(wName, 0))
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dxName, 1) == tilerModel.getTensorDimVar(wName, 0))
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 1) == 1)

        return tilerModel

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        DW policy:
          - keep full channels (C)
          - weight not tiled
          - enforce W[1]==1
        """
        dyName = parseDict[cls.gradOutKey]
        dxName = parseDict[cls.gradInKey]
        wName  = parseDict[cls.weightKey]

        dyBuf = ctxt.lookup(dyName)
        dxBuf = ctxt.lookup(dxName)
        wBuf  = ctxt.lookup(wName)

        # Full channels for both dY and dX
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == dyBuf.shape[1])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dxName, 1) == dxBuf.shape[1])

        # Weight not tiled + DW second dim fixed to 1
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 0) == wBuf.shape[0])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 1) == 1)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 2) == wBuf.shape[2])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(wName, 3) == wBuf.shape[3])

        return tilerModel

    @classmethod
    def extraSerializeChecks(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> None:
        varDY = operatorRepresentation[cls.gradOutKey]
        varDX = operatorRepresentation[cls.gradInKey]
        varW  = operatorRepresentation[cls.weightKey]

        dyFull = tuple(ctxt.lookup(varDY).shape)  # (N,C,Ho,Wo)
        dxFull = tuple(ctxt.lookup(varDX).shape)  # (N,C,Hi,Wi)
        wShape = tuple(ctxt.lookup(varW).shape)   # (C,1,P,Q)

        Cin = dxFull[1]
        Cout = dyFull[1]
        if Cin != Cout:
            raise RuntimeError(f"DWConvGradX expects Cin==Cout, got Cin={Cin}, Cout={Cout}")
        if wShape[0] != Cin:
            raise RuntimeError(f"DWConvGradX expects W[0]==C, got W[0]={wShape[0]} vs C={Cin}")
        if wShape[1] != 1:
            raise RuntimeError(f"DWConvGradX expects W[1]==1, got {wShape[1]}")

    @classmethod
    def get_dy_channels(
        cls,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
        dyName: str,
        dxName: str,
        wName: str,
        dyFull: Tuple[int, int, int, int],
        dxFull: Tuple[int, int, int, int],
        wShape: Tuple[int, int, int, int],
    ) -> int:
        # DW: dY channels is C
        return dyFull[1]

class ConvGradWTileConstraintBase(TileConstraint):
    """
    Base for ConvGradW2D tiling (im2col-style):
      - tile grad_out (dY) over H/W
      - for each dY tile, derive the required input (X) tile (with kernel halo)
      - grad_weight (dW) is NOT tiled (accumulation target is full tensor)
      - unified template padding naming:
          ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
        where:
          x => H dimension  (vertical)  => top/bottom
          y => W dimension  (horizontal)=> left/right
    """

    # ---- parser/opRep keys (override if needed) ----
    dataInKey = "data_in"    # X (forward input)
    gradOutKey = "grad_out"  # dY
    weightKey = "grad_weight"     # dW (output tensor)

    # ---------------------------
    # 1) Geometrical constraints
    # ---------------------------
    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        tilerModel.addTensorDimToModel(ctxt, xName)
        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dwName)

        group = parseDict.get("group", 1)

        # X, dY are NCHW
        N_x = tilerModel.getTensorDimVar(xName, 0)
        Ci_x = tilerModel.getTensorDimVar(xName, 1)

        N_dy = tilerModel.getTensorDimVar(dyName, 0)
        Co_dy = tilerModel.getTensorDimVar(dyName, 1)

        # dW layout (standard): [C_out, C_in_per_group, P, Q]
        Co_dw = tilerModel.getTensorDimVar(dwName, 0)
        Ci_dw = tilerModel.getTensorDimVar(dwName, 1)

        # batch match
        tilerModel.addConstraint(N_x == N_dy)

        # channel relations
        tilerModel.addConstraint(Co_dy == Co_dw)
        tilerModel.addConstraint(Ci_x == Ci_dw * group)

        return tilerModel

    # -----------------------
    # 2) Policy constraints
    # -----------------------
    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Default policy:
          - keep full Cin/Cout on X and dY
          - dW output is full (no tiling) because accumulation
          - kernel dims fixed (no tiling)
          - allow H/W tiling on dY (and derived halo on X)
        """
        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        xBuf = ctxt.lookup(xName)
        dyBuf = ctxt.lookup(dyName)
        dwBuf = ctxt.lookup(dwName)

        # Full channels for inputs
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 1) == xBuf.shape[1])   # Cin
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == dyBuf.shape[1]) # Cout

        # dW is full (all dims)
        for d in range(len(dwBuf.shape)):
            tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, d) == dwBuf.shape[d])

        # dY tile spatial dims >= 1
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 2) >= 1)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 3) >= 1)

        return tilerModel

    # -----------------------------------
    # 3) Symbolic node representation
    # -----------------------------------
    @classmethod
    def constructSymbolicNodeRep(
        cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext
    ) -> Dict[str, Union[int, IntVar]]:
        """
        Template bindings (matches your new template style for ConvGradW):
          - dim_im_out_* / ch_im_out : for grad_out (dY)
          - dim_im_in_*  / ch_im_in  : for input   (X)
          - dim_kernel_* : from dW tensor
          - padding_*    : unified naming
        """
        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        symbolic = parseDict.copy()

        # dY tile
        symbolic["dim_im_out_x"] = tilerModel.getTensorDimVar(dyName, 2)  # H_out tile
        symbolic["dim_im_out_y"] = tilerModel.getTensorDimVar(dyName, 3)  # W_out tile
        symbolic["ch_im_out"] = tilerModel.getTensorDimVar(dyName, 1)     # C_out

        # X tile
        symbolic["dim_im_in_x"] = tilerModel.getTensorDimVar(xName, 2)    # H_in tile
        symbolic["dim_im_in_y"] = tilerModel.getTensorDimVar(xName, 3)    # W_in tile
        symbolic["ch_im_in"] = tilerModel.getTensorDimVar(xName, 1)       # C_in

        # Kernel dims from dW: [C_out, C_in_per_group, P, Q]
        symbolic["dim_kernel_x"] = tilerModel.getTensorDimVar(dwName, 2)  # P (H)
        symbolic["dim_kernel_y"] = tilerModel.getTensorDimVar(dwName, 3)  # Q (W)

        return symbolic

    # -------------------------------
    # helpers
    # -------------------------------
    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return -((-a) // b)

    @staticmethod
    def _floor_div(a: int, b: int) -> int:
        return a // b

    @classmethod
    def computeInputTileFromGradOutTile(
        cls,
        kernel_hw: Tuple[int, int],                 # (P, Q)
        pads: Tuple[int, int, int, int],            # (t, b, l, r)
        strides: Tuple[int, int],                   # (sh, sw)
        inputCSize: int,                            # Cin (full)
        gradOutTile: HyperRectangle,                # dY tile (N, Cout, Ho_t, Wo_t)
        inputFull: Tuple[int, int, int, int],       # X full (N, Cin, Hi, Wi)
        gradOutFull: Tuple[int, int, int, int],     # dY full (N, Cout, Ho, Wo)
    ) -> Tuple[HyperRectangle, Tuple[int, int, int, int]]:
        """
        Given dY tile offsets, compute required X tile:
          h_in in [h_out*sh - pad_top, h_out*sh - pad_top + P)
          w_in in [w_out*sw - pad_left, w_out*sw - pad_left + Q)
        """
        (nOff, _cOff, hoOff, woOff) = gradOutTile.offset
        (nSize, _cSize, hoSize, woSize) = gradOutTile.dims

        pad_top, pad_bottom, pad_left, pad_right = pads
        sh, sw = strides
        P, Q = kernel_hw

        h_in_start = hoOff * sh - pad_top
        w_in_start = woOff * sw - pad_left

        h_in_end = (hoOff + hoSize - 1) * sh - pad_top + P
        w_in_end = (woOff + woSize - 1) * sw - pad_left + Q

        # clamp to X valid range
        h_in_start_c = max(0, h_in_start)
        w_in_start_c = max(0, w_in_start)
        h_in_end_c = min(inputFull[2], h_in_end)
        w_in_end_c = min(inputFull[3], w_in_end)

        hiSize = max(1, h_in_end_c - h_in_start_c)
        wiSize = max(1, w_in_end_c - w_in_start_c)

        xTile = HyperRectangle(
            (nOff, 0, h_in_start_c, w_in_start_c),
            (nSize, inputCSize, hiSize, wiSize),
        )

        # ONNX pads apply only on boundary tiles of dY space
        Hy = gradOutFull[2]
        Wy = gradOutFull[3]

        tile_pad_top = pad_top if hoOff == 0 else 0
        tile_pad_bottom = pad_bottom if (hoOff + hoSize) == Hy else 0
        tile_pad_left = pad_left if woOff == 0 else 0
        tile_pad_right = pad_right if (woOff + woSize) == Wy else 0

        return xTile, (tile_pad_top, tile_pad_bottom, tile_pad_left, tile_pad_right)

    @classmethod
    def extraSerializeChecks(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> None:
        """Hook for DW checks etc."""
        return

    # ---------------------------------------------------
    # 4) serialize: dY tiles -> X halo tiles, dW full
    # ---------------------------------------------------
    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        cls.extraSerializeChecks(ctxt, operatorRepresentation)

        xName = operatorRepresentation[cls.dataInKey]
        dyName = operatorRepresentation[cls.gradOutKey]
        dwName = operatorRepresentation[cls.weightKey]

        pads = tuple(operatorRepresentation.get("pads", [0, 0, 0, 0]))        # (t,b,l,r)
        strides = tuple(operatorRepresentation.get("strides", [1, 1]))        # (sh,sw)

        xFull = tuple(ctxt.lookup(xName).shape)    # (N,Cin,Hi,Wi)
        dyFull = tuple(ctxt.lookup(dyName).shape)  # (N,Cout,Ho,Wo)
        dwShape = tuple(ctxt.lookup(dwName).shape) # standard: (Cout,Cin_per_group,P,Q)

        # Use the tiler-computed dY tile shape at this mem level
        # (if missing, fall back to full dy)
        try:
            dyTileShape = tilingSolution.tensorMemoryConstraints[dyName].memoryConstraints[targetMemLevel].shape
        except Exception:
            dyTileShape = dyFull

        N_tile = dyTileShape[0]
        Ho_tile_max = dyTileShape[2]
        Wo_tile_max = dyTileShape[3]

        # Generate (ho,wo) tiles covering full dY spatial dims
        Ho_full = dyFull[2]
        Wo_full = dyFull[3]

        h_tiles: List[Tuple[int, int]] = []
        w_tiles: List[Tuple[int, int]] = []

        ho = 0
        while ho < Ho_full:
            hs = min(Ho_tile_max, Ho_full - ho)
            h_tiles.append((ho, hs))
            ho += hs

        wo = 0
        while wo < Wo_full:
            ws = min(Wo_tile_max, Wo_full - wo)
            w_tiles.append((wo, ws))
            wo += ws

        # Base addrs: inputs are X + dY, output is dW
        addrNames = [cls.dataInKey, cls.gradOutKey, cls.weightKey]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames
        )

        # Unified template naming
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

        Cin_full = xFull[1]
        Cout_full = dyFull[1]

        # dW is full cube (accumulation target)
        fullDW = HyperRectangle((0, 0, 0, 0), dwShape)

        inputLoadSchedule = []
        outputLoadSchedule = []

        # Build tiles
        for hoOff, hoSz in h_tiles:
            for woOff, woSz in w_tiles:
                dyTile = HyperRectangle(
                    (0, 0, hoOff, woOff),
                    (N_tile, Cout_full, hoSz, woSz),
                )

                xTile, (tpt, tpb, tpl, tpr) = cls.computeInputTileFromGradOutTile(
                    kernel_hw=(dwShape[2], dwShape[3]),
                    pads=pads,
                    strides=strides,
                    inputCSize=Cin_full,
                    gradOutTile=dyTile,
                    inputFull=xFull,
                    gradOutFull=dyFull,
                )

                # dims (x=H, y=W)
                replacements["dim_im_in_x"].append(xTile.dims[2])
                replacements["dim_im_in_y"].append(xTile.dims[3])
                replacements["dim_im_out_x"].append(dyTile.dims[2])
                replacements["dim_im_out_y"].append(dyTile.dims[3])

                replacements["ch_im_in"].append(Cin_full)
                replacements["ch_im_out"].append(Cout_full)

                # ONNX pads (t,b,l,r) -> unified naming:
                # padding_x_* : H dimension => top/bottom
                # padding_y_* : W dimension => left/right
                replacements["padding_y_top"].append(tpl)      # W left
                replacements["padding_y_bottom"].append(tpr)   # W right
                replacements["padding_x_left"].append(tpt)     # H top
                replacements["padding_x_right"].append(tpb)    # H bottom

                inputLoadSchedule.append({cls.dataInKey: xTile, cls.gradOutKey: dyTile})
                outputLoadSchedule.append({cls.weightKey: fullDW})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)
        return variableReplacementSchedule, tilingSchedule


class ConvGradW2DTileConstraint(ConvGradWTileConstraintBase):
    """Standard ConvGradW2D (non-depthwise)."""
    pass

class PWConvGradWTileConstraint(ConvGradWTileConstraintBase):
    """Pointwise ConvGradW (1x1 kernel)."""
    pass

class DWConvGradW2DTileConstraint(ConvGradWTileConstraintBase):
    """
    Depthwise ConvGradW:
      - X:  [N, C, Hi, Wi]
      - dY: [N, C, Ho, Wo]   (Cout == Cin == C)
      - dW: [C, 1, P, Q]
    """

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        tilerModel.addTensorDimToModel(ctxt, xName)
        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dwName)

        # N match
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 0) == tilerModel.getTensorDimVar(dyName, 0))

        # DW dW layout: [C, 1, P, Q]
        C_dw = tilerModel.getTensorDimVar(dwName, 0)
        Cpg_dw = tilerModel.getTensorDimVar(dwName, 1)

        # X and dY channels are both C
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 1) == C_dw)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == C_dw)

        # Cin_per_group must be 1
        tilerModel.addConstraint(Cpg_dw == 1)

        return tilerModel

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # Reuse base policy but also enforce DW-specific invariants tightly
        xName = parseDict[cls.dataInKey]
        dyName = parseDict[cls.gradOutKey]
        dwName = parseDict[cls.weightKey]

        xBuf = ctxt.lookup(xName)
        dyBuf = ctxt.lookup(dyName)
        dwBuf = ctxt.lookup(dwName)

        # full channels
        tilerModel.addConstraint(tilerModel.getTensorDimVar(xName, 1) == xBuf.shape[1])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 1) == dyBuf.shape[1])

        # DW invariants: Cin == Cout == dwBuf.shape[0], dwBuf.shape[1] == 1
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 0) == xBuf.shape[1])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 0) == dyBuf.shape[1])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, 1) == 1)

        # dW full (no tiling)
        for d in range(len(dwBuf.shape)):
            tilerModel.addConstraint(tilerModel.getTensorDimVar(dwName, d) == dwBuf.shape[d])

        # dY tile spatial dims >= 1
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 2) >= 1)
        tilerModel.addConstraint(tilerModel.getTensorDimVar(dyName, 3) >= 1)

        return tilerModel

    @classmethod
    def extraSerializeChecks(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> None:
        xName = operatorRepresentation[cls.dataInKey]
        dyName = operatorRepresentation[cls.gradOutKey]
        dwName = operatorRepresentation[cls.weightKey]

        xFull = tuple(ctxt.lookup(xName).shape)
        dyFull = tuple(ctxt.lookup(dyName).shape)
        dwShape = tuple(ctxt.lookup(dwName).shape)

        Cin = xFull[1]
        Cout = dyFull[1]
        assert Cin == Cout, f"DWConvGradW expects Cin==Cout, got Cin={Cin}, Cout={Cout}"
        assert dwShape[0] == Cin, f"DWConvGradW expects dW[0]==C, got dW[0]={dwShape[0]} vs C={Cin}"
        assert dwShape[1] == 1, f"DWConvGradW expects dW[1]==1, got dW[1]={dwShape[1]}"
        return
