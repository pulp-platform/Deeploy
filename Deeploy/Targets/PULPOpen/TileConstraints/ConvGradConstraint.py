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


class ConvGradX2DHWTileConstraint(TileConstraint):

    # ---------------------------
    # 1) Geometrical constraints
    # ---------------------------
    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        dyName = parseDict["data_in"]    # dY
        dxName = parseDict["data_out"]   # dX
        wName  = parseDict["weight"]     # W

        dyBuf = ctxt.lookup(dyName)
        dxBuf = ctxt.lookup(dxName)
        wBuf  = ctxt.lookup(wName)

        # Add tensor dims to model
        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dxName)
        tilerModel.addTensorDimToModel(ctxt, wName)

        group   = parseDict.get("group", 1)
        pads    = parseDict.get("pads", [0, 0, 0, 0])       # [top,bottom,left,right]
        strides = parseDict.get("strides", [1, 1])          # [sh, sw]
        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides

        # NCHW dims
        N_dy  = tilerModel.getTensorDimVar(dyName, 0)
        Co_dy = tilerModel.getTensorDimVar(dyName, 1)
        Ho_dy = tilerModel.getTensorDimVar(dyName, 2)
        Wo_dy = tilerModel.getTensorDimVar(dyName, 3)

        N_dx  = tilerModel.getTensorDimVar(dxName, 0)
        Ci_dx = tilerModel.getTensorDimVar(dxName, 1)
        Hi_dx = tilerModel.getTensorDimVar(dxName, 2)
        Wi_dx = tilerModel.getTensorDimVar(dxName, 3)

        # Weight dims: [C_out, C_in, P, Q]
        wCo = tilerModel.getTensorDimVar(wName, 0)
        wCi = tilerModel.getTensorDimVar(wName, 1)
        P   = tilerModel.getTensorDimVar(wName, 2)
        Q   = tilerModel.getTensorDimVar(wName, 3)

        # Batch match
        tilerModel.addConstraint(N_dy == N_dx)

        # Channel relations
        tilerModel.addConstraint(Co_dy == wCo)
        tilerModel.addConstraint(Ci_dx == wCi * group)
        return tilerModel


    # -----------------------
    # 2) Policy constraints
    # -----------------------
    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Policy:
        - Keep full channels (C_in, C_out)
        - Weight not tiled
        - Enforce dy tile capacity large enough for dx->dy halo (worst-case)
        """

        dyName = parseDict["data_in"]
        dxName = parseDict["data_out"]
        wName  = parseDict["weight"]

        dyBuf = ctxt.lookup(dyName)
        dxBuf = ctxt.lookup(dxName)
        wBuf  = ctxt.lookup(wName)

        group   = parseDict.get("group", 1)
        pads    = parseDict.get("pads", [0, 0, 0, 0])
        strides = parseDict.get("strides", [1, 1])
        stride_h, stride_w = strides

        # NCHW vars
        Co_dy = tilerModel.getTensorDimVar(dyName, 1)
        Ho_dy = tilerModel.getTensorDimVar(dyName, 2)
        Wo_dy = tilerModel.getTensorDimVar(dyName, 3)

        Ci_dx = tilerModel.getTensorDimVar(dxName, 1)
        Hi_dx = tilerModel.getTensorDimVar(dxName, 2)
        Wi_dx = tilerModel.getTensorDimVar(dxName, 3)

        # Weight vars: [C_out, C_in, P, Q]
        wCo = tilerModel.getTensorDimVar(wName, 0)
        wCi = tilerModel.getTensorDimVar(wName, 1)
        P   = tilerModel.getTensorDimVar(wName, 2)
        Q   = tilerModel.getTensorDimVar(wName, 3)

        # --- Full channels (because kernel memset(dX) internally) ---
        tilerModel.addConstraint(Co_dy == dyBuf.shape[1])                 # full C_out
        tilerModel.addConstraint(Ci_dx == dxBuf.shape[1])                 # full C_in

        # --- Weight not tiled ---
        tilerModel.addConstraint(wCo == wBuf.shape[0])
        tilerModel.addConstraint(wCi == wBuf.shape[1])
        tilerModel.addConstraint(P   == wBuf.shape[2])
        tilerModel.addConstraint(Q   == wBuf.shape[3])

        # --- dx->dy halo bounds (worst-case, DMA-friendly) ---
        # For stride=1:
        #   dy_h <= dx_h + (P-1)
        #   dy_w <= dx_w + (Q-1)
        tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                "dim_im_out_x",
                                                Hi_dx,
                                                P)
        

        tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                "dim_im_out_y",
                                                Wi_dx,
                                                Q)

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
        """
        IMPORTANT: You said dim_*_x = H and dim_*_y = W.
        So we bind:
          dim_im_in_x  -> H_in
          dim_im_in_y  -> W_in
          dim_im_out_x -> H_out
          dim_im_out_y -> W_out
          dim_kernel_x -> P
          dim_kernel_y -> Q
        """
        dyName = parseDict["data_in"]
        dxName = parseDict["data_out"]
        wName  = parseDict["weight"]

        symbolic = parseDict.copy()

        # dY (gradOut)
        symbolic["dim_im_out_x"] = tilerModel.getTensorDimVar(dyName, 2)  # H_out
        symbolic["dim_im_out_y"] = tilerModel.getTensorDimVar(dyName, 3)  # W_out
        symbolic["ch_im_out"]    = tilerModel.getTensorDimVar(dyName, 1)  # C_out

        # dX (gradIn)
        symbolic["dim_im_in_x"] = tilerModel.getTensorDimVar(dxName, 2)   # H_in
        symbolic["dim_im_in_y"] = tilerModel.getTensorDimVar(dxName, 3)   # W_in
        symbolic["ch_im_in"]    = tilerModel.getTensorDimVar(dxName, 1)   # C_in

        # Weight [C_out, C_in, P, Q]
        symbolic["dim_kernel_x"] = tilerModel.getTensorDimVar(wName, 2)   # P (kernel_h)
        symbolic["dim_kernel_y"] = tilerModel.getTensorDimVar(wName, 3)   # Q (kernel_w)

        # Initialize offset fields (will be filled in serializeTilingSolution)
        symbolic["offset_grad_in_h"] = 0
        symbolic["offset_grad_in_w"] = 0
        symbolic["offset_grad_out_h"] = 0
        symbolic["offset_grad_out_w"] = 0

        return symbolic

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        # b > 0
        return -((-a) // b)

    @staticmethod
    def _floor_div(a: int, b: int) -> int:
        # b > 0
        return a // b

    @classmethod
    def computeDyCubeFromDxTile(
        cls,
        dxTile: HyperRectangle,                 # (N,Cin,Hx,Wx)
        dxFull: Tuple[int,int,int,int],         # full dX
        dyFull: Tuple[int,int,int,int],         # full dY
        P: int, Q: int,                         # kernel
        pads: Tuple[int,int,int,int],           # (t,b,l,r)
        strides: Tuple[int,int],                # (sh, sw)
        dyC: int,                               # full Cout
        dxAbsOff: Tuple[int,int,int,int],        # absolute offset for boundary pads decision
    ) -> Tuple[HyperRectangle, Tuple[int,int,int,int]]:

        (nOff, _cOff, hxOff_rel, wxOff_rel) = dxTile.offset
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
                f"dxTile {dxTile.offset}/{dxTile.dims} produces empty dy range: "
                f"oy[{oy0},{oy1}], ox[{ox0},{ox1}] (Hy={Hy},Wy={Wy}, P={P},Q={Q}, pads={pads}, strides={strides})"
            )

        dyH = oy1 - oy0 + 1
        dyW = ox1 - ox0 + 1

        dyCube = HyperRectangle(
            (nOff, 0, oy0, ox0),        # dy: (N, C_out, H, W)
            (nSize, dyC, dyH, dyW)
        )

        tile_pad_top    = pad_top    if oy0 == 0 else 0
        tile_pad_bottom = pad_bottom if (oy0 + dyH) == Hy else 0
        tile_pad_left   = pad_left   if ox0 == 0 else 0
        tile_pad_right  = pad_right  if (ox0 + dyW) == Wy else 0

        return dyCube, (tile_pad_top, tile_pad_bottom, tile_pad_left, tile_pad_right)


    # ---------------------------------------------------------
    # 4) Helper: dX tile -> required dY tile (dx -> dy halo)
    # ---------------------------------------------------------
    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        # ---- resolve names early ----
        varDY = operatorRepresentation["data_in"]    # dY
        varW  = operatorRepresentation["weight"]     # W
        varDX = operatorRepresentation["data_out"]   # dX

        group   = operatorRepresentation.get("group", 1)
        pads    = operatorRepresentation.get("pads", [0,0,0,0])          # [t,b,l,r]
        strides = operatorRepresentation.get("strides", [1,1])           # [sh,sw]
        pad_top, pad_bottom, pad_left, pad_right = pads
        sh, sw = strides


        dyFull = tuple(ctxt.lookup(varDY).shape)    # (N,Cout,H,W)
        dxFull = tuple(ctxt.lookup(varDX).shape)    # (N,Cin,H,W)
        wShape = tuple(ctxt.lookup(varW).shape)     # (Cout,Cin,P,Q)
        Cout, Cin, P, Q = wShape

        outputCubes = [c.rectangle for c in absoluteOutputCubes]

        print("\n========== ConvGradX serializeTilingSolution (dx->dy, stride=1) ==========")
        print(f"varDX={varDX}, full={dxFull} | varDY={varDY}, full={dyFull} | varW={varW}, shape={wShape}")
        print(f"group={group}, pads[t,b,l,r]={pads}, strides={strides}")
        print(f"num dX tiles={len(outputCubes)}")
        print("=========================================================================\n")

        addrNames = ["data_in", "weight", "data_out"]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames
        )
        print(f"[DEBUG] inputBaseOffsets={inputBaseOffsets}")
        print(f"[DEBUG] outputBaseOffsets={outputBaseOffsets}")

        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],   # H_in  (your convention: x=H)
            "dim_im_in_y": [],   # W_in
            "dim_im_out_x": [],  # H_out
            "dim_im_out_y": [],  # W_out
            "ch_im_in": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": [],
            "offset_grad_in_h": [],
            "offset_grad_in_w": [],
            "offset_grad_out_h": [],
            "offset_grad_out_w": []
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

        maxDyH = 0
        maxDyW = 0

        for idx, dxCube in enumerate(outputCubes):
            abs_obj = absoluteOutputCubes[idx]
            abs_off = getattr(abs_obj, "absoluteOffset", None)
            if abs_off is None:
                abs_off = getattr(abs_obj, "absolute_offset", None)
            if abs_off is None:
                abs_off = dxCube.offset

            print(f"\n--- Tile {idx} ---")
            print(f"dX offset={dxCube.offset}, dims={dxCube.dims}, abs_off={abs_off}")

            (_, Cin_tile, Hx, Wx) = dxCube.dims
            if Cin_tile != dxFull[1]:
                raise RuntimeError(f"dX tile Cin must be full ({dxFull[1]}), got {Cin_tile}")

            dyCube, (tpt, tpb, tpl, tpr) = cls.computeDyCubeFromDxTile(
                dxTile=dxCube,
                dxFull=dxFull,
                dyFull=dyFull,
                P=P, Q=Q,
                pads=(pad_top, pad_bottom, pad_left, pad_right),
                strides=(sh, sw),
                dyC=Cout,
                dxAbsOff=abs_off,
            )

            print(f"-> dY offset={dyCube.offset}, dims={dyCube.dims}")
            print(f"-> tile pads(top,bottom,left,right)=({tpt},{tpb},{tpl},{tpr})")

            maxDyH = max(maxDyH, dyCube.dims[2])
            maxDyW = max(maxDyW, dyCube.dims[3])

            # replacements (x=H, y=W)
            replacements["dim_im_in_x"].append(Hx)
            replacements["dim_im_in_y"].append(Wx)
            replacements["dim_im_out_x"].append(dyCube.dims[2])
            replacements["dim_im_out_y"].append(dyCube.dims[3])

            replacements["ch_im_in"].append(dxFull[1])
            replacements["ch_im_out"].append(Cout)

            # IMPORTANT: Swap padding to match dimension convention (x=H, y=W)
            # padding_x_* is for H dimension (vertical), padding_y_* is for W dimension (horizontal)
            replacements["padding_x_left"].append(tpt)     # top (H) → x_left
            replacements["padding_x_right"].append(tpb)    # bottom (H) → x_right
            replacements["padding_y_top"].append(tpl)      # left (W) → y_top
            replacements["padding_y_bottom"].append(tpr)   # right (W) → y_bottom

            # Offset for grad_in (dX) - H and W dimensions (NCHW: [N, C, H, W])
            replacements["offset_grad_in_h"].append(abs_off[2])   # H offset
            replacements["offset_grad_in_w"].append(abs_off[3])   # W offset

            # Offset for grad_out (dY) - H and W dimensions
            replacements["offset_grad_out_h"].append(dyCube.offset[2])  # H offset
            replacements["offset_grad_out_w"].append(dyCube.offset[3])  # W offset

            # schedules
            inputDyCubes.append(dyCube)
            inputWCubes.append(HyperRectangle((0, 0, 0, 0), (Cout, Cin, P, Q)))

            print("-> kernel args check:")
            print(f"   call(H_out={dyCube.dims[2]}, W_out={dyCube.dims[3]}, C_out={Cout}, "
                f"C_in={dxFull[1]}, P={P}, Q={Q}, H_in={Hx}, W_in={Wx})")

        print("\n[DEBUG] max dY tile H,W =", (maxDyH, maxDyW))
        print("==========================================================================\n")

        inputLoadSchedule = [{"data_in": dy, "weight": w} for dy, w in zip(inputDyCubes, inputWCubes)]
        outputLoadSchedule = [{"data_out": dx} for dx in outputCubes]

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)
        return variableReplacementSchedule, tilingSchedule


class ConvGradW2DTileConstraint(TileConstraint):

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
        pads = parseDict["pads"]
        strides = parseDict["strides"]

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

        # --- Input channels must be full (required for im2col algorithm) ---
        tilerModel.addConstraint(Ci_x == parseDict["ch_im_in"])
        tilerModel.addConstraint(Co_dy == parseDict["ch_im_out"])

        # --- Compute effective input spatial dimensions (similar to Conv2D) ---
        # Account for padding when at full dimension, and kernel overlap when tiled
        effectiveInputHeight = Hi_x + ((pads[0] + pads[1]) * (Hi_x == inputBuf.shape[2])) - (
            (P - 1) * (Hi_x != inputBuf.shape[2]))
        effectiveInputWidth = Wi_x + ((pads[2] + pads[3]) * (Wi_x == inputBuf.shape[3])) - (
            (Q - 1) * (Wi_x != inputBuf.shape[3]))

        # --- Require minimum spatial dimensions to be at least kernel size ---
        tilerModel.addConstraint(effectiveInputHeight >= parseDict["dim_kernel_x"])
        tilerModel.addConstraint(effectiveInputWidth >= parseDict["dim_kernel_y"])

        # --- Ensure tiles are compatible with stride ---
        tilerModel.addConstraint((effectiveInputHeight % strides[0]) == 0)
        tilerModel.addConstraint((effectiveInputWidth % strides[1]) == 0)

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
        ConvGradW with H/W tiling (im2col approach):
        - Tile grad_out (dY) over spatial H/W dimensions
        - For each grad_out tile, compute required input (X) tile with overlap
        - All tiles accumulate into full grad_weight (dW)

        Similar to Conv2D im2col pattern.
        """

        # Extract output cubes - for ConvGradW, these represent grad_weight (always full)
        # But we create multiple cubes for different spatial computation tiles
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        varInput   = operatorRepresentation["data_in"]    # X
        varGradOut = operatorRepresentation["grad_out"]   # dY
        varGradW   = operatorRepresentation["weight"]     # dW (output)

        group   = operatorRepresentation["group"]
        pads    = operatorRepresentation["pads"]         # [t,b,l,r]
        strides = operatorRepresentation["strides"]      # [sh, sw]
        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides

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

        # For ConvGradW with H/W tiling:
        # - Output (grad_weight) is always full size (for accumulation)
        # - We tile the INPUTS (X and grad_out) spatially
        # - Extract tile dimensions from tilingSolution (computed by framework)

        # Get the tile dimensions from the memory constraint solution
        # The framework computes optimal tile sizes for L1 memory
        gradOutTileDims = tilingSolution.tensorMemoryConstraints[varGradOut].memoryConstraints[targetMemLevel].shape
        inputTileDims = tilingSolution.tensorMemoryConstraints[varInput].memoryConstraints[targetMemLevel].shape

        # Extract spatial tile sizes
        # grad_out: (N, C_out, H_out_tile, W_out_tile)
        N_tile = gradOutTileDims[0]
        H_out_tile = gradOutTileDims[2]
        W_out_tile = gradOutTileDims[3]

        # Compute how many tiles needed to cover full spatial dimensions
        # For im2col with overlap, we need to compute tile offsets carefully
        N_full, C_out_full, H_out_full, W_out_full = fullGradOutDims
        N_in_full, C_in_full, H_in_full, W_in_full = fullInputDims

        # Generate tiles to cover the full grad_out spatial dimensions
        # Each tile may overlap with neighbors due to kernel receptive field
        h_tiles = []
        w_tiles = []

        # Compute H dimension tiles
        h_offset = 0
        while h_offset < H_out_full:
            h_size = min(H_out_tile, H_out_full - h_offset)
            h_tiles.append((h_offset, h_size))
            h_offset += h_size

        # Compute W dimension tiles
        w_offset = 0
        while w_offset < W_out_full:
            w_size = min(W_out_tile, W_out_full - w_offset)
            w_tiles.append((w_offset, w_size))
            w_offset += w_size

        # Create tiles for all H/W combinations
        for h_off, h_sz in h_tiles:
            for w_off, w_sz in w_tiles:
                # Create grad_out tile for this spatial region
                gradOutCube = HyperRectangle(
                    (0, 0, h_off, w_off),
                    (N_tile, C_out_full, h_sz, w_sz)
                )

                # Compute corresponding input (X) tile with overlap
                # This uses im2col logic: input needs extra space for kernel receptive field
                inputCube, pad_tuple = cls.computeInputTileFromGradOutTile(
                    kernelShape=(weightP, weightQ),
                    pads=tuple(pads),
                    strides=(stride_h, stride_w),
                    inputCSize=C_in_full,  # Full C_in (im2col requirement)
                    gradOutTile=gradOutCube,
                    inputDims=fullInputDims,
                    gradOutDims=fullGradOutDims,
                )

                tilepad_top, tilepad_bottom, tilepad_left, tilepad_right = pad_tuple

                # Store replacements for this tile (im2col kernel parameters)
                replacements["dim_im_in_x"].append(inputCube.dims[2])      # H_in tile
                replacements["dim_im_in_y"].append(inputCube.dims[3])      # W_in tile
                replacements["dim_im_out_x"].append(h_sz)                  # H_out tile
                replacements["dim_im_out_y"].append(w_sz)                  # W_out tile
                replacements["ch_im_in"].append(C_in_full)                 # Full C_in
                replacements["ch_im_out"].append(C_out_full)               # Full C_out

                # Tile-specific padding (edge tiles have padding, interior tiles don't)
                replacements["padding_y_top"].append(tilepad_top)
                replacements["padding_y_bottom"].append(tilepad_bottom)
                replacements["padding_x_left"].append(tilepad_left)
                replacements["padding_x_right"].append(tilepad_right)

                inputXCubes.append(inputCube)
                inputGradOutCubes.append(gradOutCube)

        # Build load schedules
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Each input tile pair (X, grad_out) computes a partial gradient
        for x_cube, dy_cube in zip(inputXCubes, inputGradOutCubes):
            inputLoadSchedule.append({"data_in": x_cube, "grad_out": dy_cube})

        # All tiles accumulate into the same full grad_weight
        # Create one output entry per input tile (all point to same full grad_weight)
        fullGradWeightCube = HyperRectangle((0, 0, 0, 0), (weightCo, weightCi, weightP, weightQ))
        for _ in range(len(inputXCubes)):
            outputLoadSchedule.append({"weight": fullGradWeightCube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class ConvGradX2DCTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

    #     return tilerModel
        dyName = parseDict["data_in"]    # dY
        dxName = parseDict["data_out"]   # dX
        wName  = parseDict["weight"]     # W

        # Add tensor dims to model
        tilerModel.addTensorDimToModel(ctxt, dyName)
        tilerModel.addTensorDimToModel(ctxt, dxName)
        tilerModel.addTensorDimToModel(ctxt, wName)

        group   = parseDict.get("group", 1)

        # NCHW dims
        N_dy  = tilerModel.getTensorDimVar(dyName, 0)
        Co_dy = tilerModel.getTensorDimVar(dyName, 1)

        N_dx  = tilerModel.getTensorDimVar(dxName, 0)
        Ci_dx = tilerModel.getTensorDimVar(dxName, 1)

        wCo = tilerModel.getTensorDimVar(wName, 0)
        wCi = tilerModel.getTensorDimVar(wName, 1)

        # Batch match
        tilerModel.addConstraint(N_dy == N_dx)

        # Channel relations
        tilerModel.addConstraint(Co_dy == wCo)
        tilerModel.addConstraint(Ci_dx == wCi * group)
        return tilerModel


    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']
        numDims = len(ctxt.lookup(inputBufferName).shape)

        for idx in range(numDims):
            if idx != numDims - 1:  # RW: Keep all dimensions except C (index 1 in NCHW) fixed
                tilerModel.addConstraint(
                    tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx) == ctxt.lookup(
                        outputBufferName).shape[idx])
                tilerModel.addConstraint(
                    tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx) == ctxt.lookup(
                        inputBufferName).shape[idx])

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        inputInCubes = []
        replacementTypes = {}
        replacements: Dict[str, List[int]] = {}

        numDims = len(ctxt.lookup(operatorRepresentation['data_in']).shape)
        replacementTypes["ch_im_in"] = PointerClass(uint16_t)
        replacements["ch_im_in"] = []

        input_shape = ctxt.lookup(operatorRepresentation['data_in']).shape
        output_shape = ctxt.lookup(operatorRepresentation['data_out']).shape

        for cube in outputCubes:
            input_offset = list(cube.offset)
            input_dims = list(input_shape)
            input_dims[-1] = cube.dims[-1]
            InCube = HyperRectangle(tuple(input_offset), tuple(input_dims))
            inputInCubes.append(InCube)

            replacements["ch_im_in"].append(input_dims[-1])

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a in inputInCubes:
            inputLoadSchedule.append({"data_in": a})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

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
