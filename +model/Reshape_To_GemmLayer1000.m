classdef Reshape_To_GemmLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        x_conv1_Constant_2_o
        x_conv1_ConstantOf_1
        onnx__Tile_18
        x_conv1_Constant_6_o
        x_conv1_Constant_8_o
        x_conv1_Constant_10_
        onnx__MatMul_266
        x_conv2_Constant_3_o
        x_conv2_Constant_4_o
        x_conv2_ConstantOf_3
        x_conv2_Constant_9_o
        x_conv2_Constant_11_
        onnx__MatMul_270
        x_conv3_Constant_3_o
        x_conv3_Constant_4_o
        x_conv3_ConstantOf_3
        x_conv3_Constant_9_o
        x_conv3_Constant_11_
        onnx__MatMul_273
        x_Constant_2_output_
        lin_weight
        lin_bias
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end




    methods
        function this = Reshape_To_GemmLayer1000(name)
            this.Name = name;
            this.NumInputs = 6;
            this.OutputNames = {'output'};
        end

        function [output] = predict(this, edge_index, x, batch, edge_indexNumDims, xNumDims, batchNumDims)
            if isdlarray(edge_index)
                edge_index = stripdims(edge_index);
            end
            if isdlarray(x)
                x = stripdims(x);
            end
            if isdlarray(batch)
                batch = stripdims(batch);
            end
            edge_indexNumDims = numel(edge_indexNumDims);
            xNumDims = numel(xNumDims);
            batchNumDims = numel(batchNumDims);
            edge_index = model.ops.permuteInputVar(edge_index, ['as-is'], 2);
            x = model.ops.permuteInputVar(x, ['as-is'], 2);
            batch = model.ops.permuteInputVar(batch, ['as-is'], 1);

            [output, outputNumDims] = Reshape_To_GemmGraph1000(this, edge_index, x, batch, edge_indexNumDims, xNumDims, batchNumDims, false);
            output = model.ops.permuteOutputVar(output, ['as-is'], 2);

            output = dlarray(single(output), repmat('U', 1, max(2, outputNumDims)));
        end

        function [output] = forward(this, edge_index, x, batch, edge_indexNumDims, xNumDims, batchNumDims)
            if isdlarray(edge_index)
                edge_index = stripdims(edge_index);
            end
            if isdlarray(x)
                x = stripdims(x);
            end
            if isdlarray(batch)
                batch = stripdims(batch);
            end
            edge_indexNumDims = numel(edge_indexNumDims);
            xNumDims = numel(xNumDims);
            batchNumDims = numel(batchNumDims);
            edge_index = model.ops.permuteInputVar(edge_index, ['as-is'], 2);
            x = model.ops.permuteInputVar(x, ['as-is'], 2);
            batch = model.ops.permuteInputVar(batch, ['as-is'], 1);

            [output, outputNumDims] = Reshape_To_GemmGraph1000(this, edge_index, x, batch, edge_indexNumDims, xNumDims, batchNumDims, true);
            output = model.ops.permuteOutputVar(output, ['as-is'], 2);

            output = dlarray(single(output), repmat('U', 1, max(2, outputNumDims)));
        end

        function [output, outputNumDims1037] = Reshape_To_GemmGraph1000(this, edge_index, x, batch, edge_indexNumDims, xNumDims, batchNumDims, Training)

            % Execute the operators:
            % Gather:
            [x_conv1_Gather_outpu, x_conv1_Gather_outpuNumDims] = model.ops.onnxGather(edge_index, this.Vars.x_conv1_Constant_out, 0, edge_indexNumDims, this.NumDims.x_conv1_Constant_out);

            % Gather:
            [x_conv1_Gather_1_out, x_conv1_Gather_1_outNumDims] = model.ops.onnxGather(edge_index, this.Vars.x_conv1_Constant_1_o, 0, edge_indexNumDims, this.NumDims.x_conv1_Constant_1_o);

            % Equal:
            x_conv1_Equal_output = x_conv1_Gather_outpu == x_conv1_Gather_1_out;
            x_conv1_Equal_outputNumDims = max(x_conv1_Gather_outpuNumDims, x_conv1_Gather_1_outNumDims);

            % Not:
            x_conv1_Not_output_0 = not(x_conv1_Equal_output);
            x_conv1_Not_output_0NumDims = x_conv1_Equal_outputNumDims;

            % Expand:
            [shape, x_conv1_Expand_outpuNumDims] = model.ops.prepareExpandArgs(this.x_conv1_ConstantOf_1);
            x_conv1_Expand_outpu = this.x_conv1_Constant_2_o + zeros(shape);

            % Tile:
            [sz, x_conv1_Tile_output_NumDims] = model.ops.prepareTileArgs(this.onnx__Tile_18);
            x_conv1_Tile_output_ = repmat(x_conv1_Expand_outpu, sz);

            % NonZero:
            [x_conv1_NonZero_outp, x_conv1_NonZero_outpNumDims] = model.ops.onnxNonZero(x_conv1_Not_output_0, x_conv1_Not_output_0NumDims);

            % Transpose:
            [perm, x_conv1_Transpose_ouNumDims] = model.ops.prepareTransposeArgs(this.Vars.TransposePerm1001, x_conv1_NonZero_outpNumDims);
            if isempty(perm)
                x_conv1_Transpose_ou = x_conv1_NonZero_outp;
            else
                x_conv1_Transpose_ou = permute(x_conv1_NonZero_outp, perm);
            end

            % Squeeze:
            [x_conv1_Squeeze_outp, x_conv1_Squeeze_outpNumDims] = model.ops.onnxSqueeze(x_conv1_Transpose_ou, this.Vars.SqueezeAxes1002, x_conv1_Transpose_ouNumDims);

            % Gather:
            [x_conv1_Gather_2_out, x_conv1_Gather_2_outNumDims] = model.ops.onnxGather(edge_index, x_conv1_Squeeze_outp, 1, edge_indexNumDims, x_conv1_Squeeze_outpNumDims);

            % Concat:
            [x_conv1_Concat_outpu, x_conv1_Concat_outpuNumDims] = model.ops.onnxConcat(1, {x_conv1_Gather_2_out, x_conv1_Tile_output_}, [x_conv1_Gather_2_outNumDims, x_conv1_Tile_output_NumDims]);

            % Shape:
            [x_conv1_Shape_output, x_conv1_Shape_outputNumDims] = model.ops.onnxShape(x_conv1_Concat_outpu, x_conv1_Concat_outpuNumDims, 0, x_conv1_Concat_outpuNumDims+1);

            % Gather:
            [x_conv1_Gather_3_out, x_conv1_Gather_3_outNumDims] = model.ops.onnxGather(x_conv1_Shape_output, this.Vars.x_conv1_Constant_4_o, 0, x_conv1_Shape_outputNumDims, this.NumDims.x_conv1_Constant_4_o);

            % Unsqueeze:
            [shape, x_conv1_Unsqueeze_ouNumDims] = model.ops.prepareUnsqueezeArgs(x_conv1_Gather_3_out, this.Vars.UnsqueezeAxes1003, x_conv1_Gather_3_outNumDims);
            x_conv1_Unsqueeze_ou = reshape(x_conv1_Gather_3_out, shape);

            % Concat:
            [x_conv1_Concat_1_out, x_conv1_Concat_1_outNumDims] = model.ops.onnxConcat(0, {x_conv1_Unsqueeze_ou}, [x_conv1_Unsqueeze_ouNumDims]);

            % ConstantOfShape:
            [x_conv1_ConstantOfSh, x_conv1_ConstantOfShNumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1004, x_conv1_Concat_1_out);

            % Gather:
            [x_conv1_Gather_4_out, x_conv1_Gather_4_outNumDims] = model.ops.onnxGather(x_conv1_Concat_outpu, this.Vars.x_conv1_Constant_out, 0, x_conv1_Concat_outpuNumDims, this.NumDims.x_conv1_Constant_out);

            % Gather:
            [x_conv1_Gather_5_out, x_conv1_Gather_5_outNumDims] = model.ops.onnxGather(x_conv1_Concat_outpu, this.Vars.x_conv1_Constant_1_o, 0, x_conv1_Concat_outpuNumDims, this.NumDims.x_conv1_Constant_1_o);

            % Reshape:
            [shape, x_conv1_Reshape_outpNumDims] = model.ops.prepareReshapeArgs(x_conv1_Gather_5_out, this.Vars.x_conv1_Constant_5_o, x_conv1_Gather_5_outNumDims, 0);
            x_conv1_Reshape_outp = reshape(x_conv1_Gather_5_out, shape{:});

            % Shape:
            [x_conv1_Shape_1_outp, x_conv1_Shape_1_outpNumDims] = model.ops.onnxShape(x_conv1_ConstantOfSh, x_conv1_ConstantOfShNumDims, 0, x_conv1_ConstantOfShNumDims+1);

            % Expand:
            [shape, x_conv1_Expand_1_outNumDims] = model.ops.prepareExpandArgs(x_conv1_Shape_1_outp);
            x_conv1_Expand_1_out = x_conv1_Reshape_outp + zeros(shape);

            % ScatterElements:
            [x_conv1_ScatterEleme, x_conv1_ScatterElemeNumDims] = model.ops.onnxScatterElements(this.x_conv1_Constant_6_o, x_conv1_Expand_1_out, x_conv1_ConstantOfSh, 0, "none", this.NumDims.x_conv1_Constant_6_o);

            % Add:
            x_conv1_Add_output_0 = this.Vars.x_conv1_Constant_7_o + x_conv1_ScatterEleme;
            x_conv1_Add_output_0NumDims = max(this.NumDims.x_conv1_Constant_7_o, x_conv1_ScatterElemeNumDims);

            % Pow:
            x_conv1_Pow_output_0 = power(x_conv1_Add_output_0, this.x_conv1_Constant_8_o);
            x_conv1_Pow_output_0NumDims = max(x_conv1_Add_output_0NumDims, this.NumDims.x_conv1_Constant_8_o);

            % Equal:
            x_conv1_Equal_1_outp = x_conv1_Pow_output_0 == this.Vars.x_conv1_Constant_9_o;
            x_conv1_Equal_1_outpNumDims = max(x_conv1_Pow_output_0NumDims, this.NumDims.x_conv1_Constant_9_o);

            % Cast:
            x_conv1_Cast_output_ = logical(x_conv1_Equal_1_outp);
            x_conv1_Cast_output_NumDims = x_conv1_Equal_1_outpNumDims;

            % Where:
            [x_conv1_Where_output, x_conv1_Where_outputNumDims] = model.ops.onnxWhere(x_conv1_Cast_output_, this.x_conv1_Constant_10_, x_conv1_Pow_output_0, x_conv1_Cast_output_NumDims, this.NumDims.x_conv1_Constant_10_, x_conv1_Pow_output_0NumDims);

            % Gather:
            [x_conv1_Gather_6_out, x_conv1_Gather_6_outNumDims] = model.ops.onnxGather(x_conv1_Where_output, x_conv1_Gather_4_out, 0, x_conv1_Where_outputNumDims, x_conv1_Gather_4_outNumDims);

            % Mul:
            x_conv1_Mul_output_0 = x_conv1_Gather_6_out .* x_conv1_ConstantOfSh;
            x_conv1_Mul_output_0NumDims = max(x_conv1_Gather_6_outNumDims, x_conv1_ConstantOfShNumDims);

            % Gather:
            [x_conv1_Gather_7_out, x_conv1_Gather_7_outNumDims] = model.ops.onnxGather(x_conv1_Where_output, x_conv1_Gather_5_out, 0, x_conv1_Where_outputNumDims, x_conv1_Gather_5_outNumDims);

            % Mul:
            x_conv1_Mul_1_output = x_conv1_Mul_output_0 .* x_conv1_Gather_7_out;
            x_conv1_Mul_1_outputNumDims = max(x_conv1_Mul_output_0NumDims, x_conv1_Gather_7_outNumDims);

            % MatMul:
            [x_conv1_lin_MatMul_o, x_conv1_lin_MatMul_oNumDims] = model.ops.onnxMatMul(x, this.onnx__MatMul_266, xNumDims, this.NumDims.onnx__MatMul_266);

            % Gather:
            [x_conv1_Gather_8_out, x_conv1_Gather_8_outNumDims] = model.ops.onnxGather(x_conv1_lin_MatMul_o, x_conv1_Gather_4_out, -2, x_conv1_lin_MatMul_oNumDims, x_conv1_Gather_4_outNumDims);

            % Reshape:
            [shape, x_conv1_Reshape_1_ouNumDims] = model.ops.prepareReshapeArgs(x_conv1_Mul_1_output, this.Vars.x_conv1_Constant_11_, x_conv1_Mul_1_outputNumDims, 0);
            x_conv1_Reshape_1_ou = reshape(x_conv1_Mul_1_output, shape{:});

            % Mul:
            x_conv1_Mul_2_output = x_conv1_Reshape_1_ou .* x_conv1_Gather_8_out;
            x_conv1_Mul_2_outputNumDims = max(x_conv1_Reshape_1_ouNumDims, x_conv1_Gather_8_outNumDims);

            % Shape:
            [x_conv1_aggr_modu_13, x_conv1_aggr_modu_13NumDims] = model.ops.onnxShape(x_conv1_Mul_2_output, x_conv1_Mul_2_outputNumDims, 0, x_conv1_Mul_2_outputNumDims+1);

            % Gather:
            [x_conv1_aggr_modu_8, x_conv1_aggr_modu_8NumDims] = model.ops.onnxGather(x_conv1_aggr_modu_13, this.Vars.x_conv1_aggr_modu_6, 0, x_conv1_aggr_modu_13NumDims, this.NumDims.x_conv1_aggr_modu_6);

            % Reshape:
            [shape, x_conv1_aggr_modu_9NumDims] = model.ops.prepareReshapeArgs(x_conv1_Gather_5_out, this.Vars.x_conv1_aggr_modu_4, x_conv1_Gather_5_outNumDims, 0);
            x_conv1_aggr_modu_9 = reshape(x_conv1_Gather_5_out, shape{:});

            % Shape:
            [x_conv1_aggr_modu_11, x_conv1_aggr_modu_11NumDims] = model.ops.onnxShape(x_conv1_Mul_2_output, x_conv1_Mul_2_outputNumDims, 0, x_conv1_Mul_2_outputNumDims+1);

            % Expand:
            [shape, x_conv1_aggr_modu_7NumDims] = model.ops.prepareExpandArgs(x_conv1_aggr_modu_11);
            x_conv1_aggr_modu_7 = x_conv1_aggr_modu_9 + zeros(shape);

            % Unsqueeze:
            [shape, x_conv1_aggr_modu_14NumDims] = model.ops.prepareUnsqueezeArgs(x_conv1_aggr_modu_8, this.Vars.UnsqueezeAxes1005, x_conv1_aggr_modu_8NumDims);
            x_conv1_aggr_modu_14 = reshape(x_conv1_aggr_modu_8, shape);

            % Concat:
            [x_conv1_aggr_modu_1, x_conv1_aggr_modu_1NumDims] = model.ops.onnxConcat(0, {this.Vars.x_conv1_aggr_modu_5, x_conv1_aggr_modu_14}, [this.NumDims.x_conv1_aggr_modu_5, x_conv1_aggr_modu_14NumDims]);

            % ConstantOfShape:
            [x_conv1_aggr_modu_3, x_conv1_aggr_modu_3NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1006, x_conv1_aggr_modu_1);

            % Shape:
            [x_conv1_aggr_modu_12, x_conv1_aggr_modu_12NumDims] = model.ops.onnxShape(x_conv1_aggr_modu_3, x_conv1_aggr_modu_3NumDims, 0, x_conv1_aggr_modu_3NumDims+1);

            % ConstantOfShape:
            [x_conv1_aggr_modu_2, x_conv1_aggr_modu_2NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1007, x_conv1_aggr_modu_12);

            % ScatterElements:
            [x_conv1_aggr_modu_10, x_conv1_aggr_modu_10NumDims] = model.ops.onnxScatterElements(x_conv1_aggr_modu_2, x_conv1_aggr_modu_7, x_conv1_Mul_2_output, 0, "none", x_conv1_aggr_modu_2NumDims);

            % Add:
            x_conv1_aggr_module_ = x_conv1_aggr_modu_3 + x_conv1_aggr_modu_10;
            x_conv1_aggr_module_NumDims = max(x_conv1_aggr_modu_3NumDims, x_conv1_aggr_modu_10NumDims);

            % Add:
            x_conv1_Add_1_output = x_conv1_aggr_module_ + this.Vars.conv1_bias;
            x_conv1_Add_1_outputNumDims = max(x_conv1_aggr_module_NumDims, this.NumDims.conv1_bias);

            % Relu:
            x_Relu_output_0 = relu(dlarray(x_conv1_Add_1_output));
            x_Relu_output_0NumDims = x_conv1_Add_1_outputNumDims;

            % Shape:
            [x_conv2_Shape_output, x_conv2_Shape_outputNumDims] = model.ops.onnxShape(x_Relu_output_0, x_Relu_output_0NumDims, 0, x_Relu_output_0NumDims+1);

            % Slice:
            [Indices, x_conv2_Slice_outputNumDims] = model.ops.prepareSliceArgs(x_conv2_Shape_output, this.Vars.x_conv2_Constant_1_o, this.Vars.x_conv2_Constant_2_o, this.Vars.x_conv2_Constant_out, '', x_conv2_Shape_outputNumDims);
            x_conv2_Slice_output = x_conv2_Shape_output(Indices{:});

            % Squeeze:
            [x_conv2_Squeeze_outp, x_conv2_Squeeze_outpNumDims] = model.ops.onnxSqueeze(x_conv2_Slice_output, this.Vars.SqueezeAxes1008, x_conv2_Slice_outputNumDims);

            % Cast:
            x_conv2_Cast_output_ = cast(int64(extractdata(x_conv2_Squeeze_outp)), 'like', x_conv2_Squeeze_outp);
            x_conv2_Cast_output_NumDims = x_conv2_Squeeze_outpNumDims;

            % Range:
            x_conv2_Range_output = this.x_conv2_Constant_3_o:this.x_conv2_Constant_4_o:x_conv2_Cast_output_-sign(this.x_conv2_Constant_4_o)';
            x_conv2_Range_outputNumDims = 1;

            % Reshape:
            [shape, x_conv2_Reshape_outpNumDims] = model.ops.prepareReshapeArgs(x_conv2_Range_output, this.Vars.x_conv2_Constant_5_o, x_conv2_Range_outputNumDims, 0);
            x_conv2_Reshape_outp = reshape(x_conv2_Range_output, shape{:});

            % Expand:
            [shape, x_conv2_Expand_outpuNumDims] = model.ops.prepareExpandArgs(this.x_conv2_ConstantOf_3);
            x_conv2_Expand_outpu = x_conv2_Reshape_outp + zeros(shape);

            % Tile:
            [sz, x_conv2_Tile_output_NumDims] = model.ops.prepareTileArgs(this.onnx__Tile_18);
            x_conv2_Tile_output_ = repmat(x_conv2_Expand_outpu, sz);

            % Concat:
            [x_conv2_Concat_outpu, x_conv2_Concat_outpuNumDims] = model.ops.onnxConcat(1, {x_conv1_Gather_2_out, x_conv2_Tile_output_}, [x_conv1_Gather_2_outNumDims, x_conv2_Tile_output_NumDims]);

            % Shape:
            [x_conv2_Shape_1_outp, x_conv2_Shape_1_outpNumDims] = model.ops.onnxShape(x_conv2_Concat_outpu, x_conv2_Concat_outpuNumDims, 0, x_conv2_Concat_outpuNumDims+1);

            % Gather:
            [x_conv2_Gather_outpu, x_conv2_Gather_outpuNumDims] = model.ops.onnxGather(x_conv2_Shape_1_outp, this.Vars.x_conv2_Constant_7_o, 0, x_conv2_Shape_1_outpNumDims, this.NumDims.x_conv2_Constant_7_o);

            % Unsqueeze:
            [shape, x_conv2_Unsqueeze_ouNumDims] = model.ops.prepareUnsqueezeArgs(x_conv2_Gather_outpu, this.Vars.UnsqueezeAxes1009, x_conv2_Gather_outpuNumDims);
            x_conv2_Unsqueeze_ou = reshape(x_conv2_Gather_outpu, shape);

            % Concat:
            [x_conv2_Concat_1_out, x_conv2_Concat_1_outNumDims] = model.ops.onnxConcat(0, {x_conv2_Unsqueeze_ou}, [x_conv2_Unsqueeze_ouNumDims]);

            % ConstantOfShape:
            [x_conv2_ConstantOfSh, x_conv2_ConstantOfShNumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1010, x_conv2_Concat_1_out);

            % Gather:
            [x_conv2_Gather_1_out, x_conv2_Gather_1_outNumDims] = model.ops.onnxGather(x_conv2_Concat_outpu, this.Vars.x_conv1_Constant_out, 0, x_conv2_Concat_outpuNumDims, this.NumDims.x_conv1_Constant_out);

            % Gather:
            [x_conv2_Gather_2_out, x_conv2_Gather_2_outNumDims] = model.ops.onnxGather(x_conv2_Concat_outpu, this.Vars.x_conv1_Constant_1_o, 0, x_conv2_Concat_outpuNumDims, this.NumDims.x_conv1_Constant_1_o);

            % Reshape:
            [shape, x_conv2_Reshape_1_ouNumDims] = model.ops.prepareReshapeArgs(x_conv2_Gather_2_out, this.Vars.x_conv2_Constant_8_o, x_conv2_Gather_2_outNumDims, 0);
            x_conv2_Reshape_1_ou = reshape(x_conv2_Gather_2_out, shape{:});

            % Shape:
            [x_conv2_Shape_2_outp, x_conv2_Shape_2_outpNumDims] = model.ops.onnxShape(x_conv2_ConstantOfSh, x_conv2_ConstantOfShNumDims, 0, x_conv2_ConstantOfShNumDims+1);

            % Expand:
            [shape, x_conv2_Expand_1_outNumDims] = model.ops.prepareExpandArgs(x_conv2_Shape_2_outp);
            x_conv2_Expand_1_out = x_conv2_Reshape_1_ou + zeros(shape);

            % Unsqueeze:
            [shape, x_conv2_Unsqueeze_1_NumDims] = model.ops.prepareUnsqueezeArgs(x_conv2_Squeeze_outp, this.Vars.UnsqueezeAxes1011, x_conv2_Squeeze_outpNumDims);
            x_conv2_Unsqueeze_1_ = reshape(x_conv2_Squeeze_outp, shape);

            % Concat:
            [x_conv2_Concat_2_out, x_conv2_Concat_2_outNumDims] = model.ops.onnxConcat(0, {x_conv2_Unsqueeze_1_}, [x_conv2_Unsqueeze_1_NumDims]);

            % ConstantOfShape:
            [x_conv2_ConstantOf_1, x_conv2_ConstantOf_1NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1012, x_conv2_Concat_2_out);

            % Shape:
            [x_conv2_Shape_3_outp, x_conv2_Shape_3_outpNumDims] = model.ops.onnxShape(x_conv2_ConstantOf_1, x_conv2_ConstantOf_1NumDims, 0, x_conv2_ConstantOf_1NumDims+1);

            % ConstantOfShape:
            [x_conv2_ConstantOf_2, x_conv2_ConstantOf_2NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1013, x_conv2_Shape_3_outp);

            % ScatterElements:
            [x_conv2_ScatterEleme, x_conv2_ScatterElemeNumDims] = model.ops.onnxScatterElements(x_conv2_ConstantOf_2, x_conv2_Expand_1_out, x_conv2_ConstantOfSh, 0, "none", x_conv2_ConstantOf_2NumDims);

            % Add:
            x_conv2_Add_output_0 = x_conv2_ConstantOf_1 + x_conv2_ScatterEleme;
            x_conv2_Add_output_0NumDims = max(x_conv2_ConstantOf_1NumDims, x_conv2_ScatterElemeNumDims);

            % Pow:
            x_conv2_Pow_output_0 = power(x_conv2_Add_output_0, this.x_conv2_Constant_9_o);
            x_conv2_Pow_output_0NumDims = max(x_conv2_Add_output_0NumDims, this.NumDims.x_conv2_Constant_9_o);

            % Equal:
            x_conv2_Equal_output = x_conv2_Pow_output_0 == this.Vars.x_conv2_Constant_10_;
            x_conv2_Equal_outputNumDims = max(x_conv2_Pow_output_0NumDims, this.NumDims.x_conv2_Constant_10_);

            % Cast:
            x_conv2_Cast_1_outpu = logical(x_conv2_Equal_output);
            x_conv2_Cast_1_outpuNumDims = x_conv2_Equal_outputNumDims;

            % Where:
            [x_conv2_Where_output, x_conv2_Where_outputNumDims] = model.ops.onnxWhere(x_conv2_Cast_1_outpu, this.x_conv2_Constant_11_, x_conv2_Pow_output_0, x_conv2_Cast_1_outpuNumDims, this.NumDims.x_conv2_Constant_11_, x_conv2_Pow_output_0NumDims);

            % Gather:
            [x_conv2_Gather_3_out, x_conv2_Gather_3_outNumDims] = model.ops.onnxGather(x_conv2_Where_output, x_conv2_Gather_1_out, 0, x_conv2_Where_outputNumDims, x_conv2_Gather_1_outNumDims);

            % Mul:
            x_conv2_Mul_output_0 = x_conv2_Gather_3_out .* x_conv2_ConstantOfSh;
            x_conv2_Mul_output_0NumDims = max(x_conv2_Gather_3_outNumDims, x_conv2_ConstantOfShNumDims);

            % Gather:
            [x_conv2_Gather_4_out, x_conv2_Gather_4_outNumDims] = model.ops.onnxGather(x_conv2_Where_output, x_conv2_Gather_2_out, 0, x_conv2_Where_outputNumDims, x_conv2_Gather_2_outNumDims);

            % Mul:
            x_conv2_Mul_1_output = x_conv2_Mul_output_0 .* x_conv2_Gather_4_out;
            x_conv2_Mul_1_outputNumDims = max(x_conv2_Mul_output_0NumDims, x_conv2_Gather_4_outNumDims);

            % MatMul:
            [x_conv2_lin_MatMul_o, x_conv2_lin_MatMul_oNumDims] = model.ops.onnxMatMul(x_Relu_output_0, this.onnx__MatMul_270, x_Relu_output_0NumDims, this.NumDims.onnx__MatMul_270);

            % Shape:
            [x_conv2_Shape_4_outp, x_conv2_Shape_4_outpNumDims] = model.ops.onnxShape(x_conv2_lin_MatMul_o, x_conv2_lin_MatMul_oNumDims, 0, x_conv2_lin_MatMul_oNumDims+1);

            % Slice:
            [Indices, x_conv2_Slice_1_outpNumDims] = model.ops.prepareSliceArgs(x_conv2_Shape_4_outp, this.Vars.x_conv2_Constant_13_, this.Vars.x_conv2_Constant_14_, this.Vars.x_conv2_Constant_12_, '', x_conv2_Shape_4_outpNumDims);
            x_conv2_Slice_1_outp = x_conv2_Shape_4_outp(Indices{:});

            % Squeeze:
            [x_conv2_Squeeze_1_ou, x_conv2_Squeeze_1_ouNumDims] = model.ops.onnxSqueeze(x_conv2_Slice_1_outp, this.Vars.SqueezeAxes1014, x_conv2_Slice_1_outpNumDims);

            % Gather:
            [x_conv2_Gather_5_out, x_conv2_Gather_5_outNumDims] = model.ops.onnxGather(x_conv2_lin_MatMul_o, x_conv2_Gather_1_out, -2, x_conv2_lin_MatMul_oNumDims, x_conv2_Gather_1_outNumDims);

            % Reshape:
            [shape, x_conv2_Reshape_2_ouNumDims] = model.ops.prepareReshapeArgs(x_conv2_Mul_1_output, this.Vars.x_conv2_Constant_15_, x_conv2_Mul_1_outputNumDims, 0);
            x_conv2_Reshape_2_ou = reshape(x_conv2_Mul_1_output, shape{:});

            % Mul:
            x_conv2_Mul_2_output = x_conv2_Reshape_2_ou .* x_conv2_Gather_5_out;
            x_conv2_Mul_2_outputNumDims = max(x_conv2_Reshape_2_ouNumDims, x_conv2_Gather_5_outNumDims);

            % Shape:
            [x_conv2_aggr_modu_12, x_conv2_aggr_modu_12NumDims] = model.ops.onnxShape(x_conv2_Mul_2_output, x_conv2_Mul_2_outputNumDims, 0, x_conv2_Mul_2_outputNumDims+1);

            % Gather:
            [x_conv2_aggr_modu_7, x_conv2_aggr_modu_7NumDims] = model.ops.onnxGather(x_conv2_aggr_modu_12, this.Vars.x_conv2_aggr_modu_5, 0, x_conv2_aggr_modu_12NumDims, this.NumDims.x_conv2_aggr_modu_5);

            % Reshape:
            [shape, x_conv2_aggr_modu_8NumDims] = model.ops.prepareReshapeArgs(x_conv2_Gather_2_out, this.Vars.x_conv2_aggr_modu_4, x_conv2_Gather_2_outNumDims, 0);
            x_conv2_aggr_modu_8 = reshape(x_conv2_Gather_2_out, shape{:});

            % Shape:
            [x_conv2_aggr_modu_10, x_conv2_aggr_modu_10NumDims] = model.ops.onnxShape(x_conv2_Mul_2_output, x_conv2_Mul_2_outputNumDims, 0, x_conv2_Mul_2_outputNumDims+1);

            % Expand:
            [shape, x_conv2_aggr_modu_6NumDims] = model.ops.prepareExpandArgs(x_conv2_aggr_modu_10);
            x_conv2_aggr_modu_6 = x_conv2_aggr_modu_8 + zeros(shape);

            % Unsqueeze:
            [shape, x_conv2_aggr_modu_14NumDims] = model.ops.prepareUnsqueezeArgs(x_conv2_Squeeze_1_ou, this.Vars.UnsqueezeAxes1015, x_conv2_Squeeze_1_ouNumDims);
            x_conv2_aggr_modu_14 = reshape(x_conv2_Squeeze_1_ou, shape);

            % Unsqueeze:
            [shape, x_conv2_aggr_modu_13NumDims] = model.ops.prepareUnsqueezeArgs(x_conv2_aggr_modu_7, this.Vars.UnsqueezeAxes1016, x_conv2_aggr_modu_7NumDims);
            x_conv2_aggr_modu_13 = reshape(x_conv2_aggr_modu_7, shape);

            % Concat:
            [x_conv2_aggr_modu_1, x_conv2_aggr_modu_1NumDims] = model.ops.onnxConcat(0, {x_conv2_aggr_modu_14, x_conv2_aggr_modu_13}, [x_conv2_aggr_modu_14NumDims, x_conv2_aggr_modu_13NumDims]);

            % ConstantOfShape:
            [x_conv2_aggr_modu_3, x_conv2_aggr_modu_3NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1017, x_conv2_aggr_modu_1);

            % Shape:
            [x_conv2_aggr_modu_11, x_conv2_aggr_modu_11NumDims] = model.ops.onnxShape(x_conv2_aggr_modu_3, x_conv2_aggr_modu_3NumDims, 0, x_conv2_aggr_modu_3NumDims+1);

            % ConstantOfShape:
            [x_conv2_aggr_modu_2, x_conv2_aggr_modu_2NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1018, x_conv2_aggr_modu_11);

            % ScatterElements:
            [x_conv2_aggr_modu_9, x_conv2_aggr_modu_9NumDims] = model.ops.onnxScatterElements(x_conv2_aggr_modu_2, x_conv2_aggr_modu_6, x_conv2_Mul_2_output, 0, "none", x_conv2_aggr_modu_2NumDims);

            % Add:
            x_conv2_aggr_module_ = x_conv2_aggr_modu_3 + x_conv2_aggr_modu_9;
            x_conv2_aggr_module_NumDims = max(x_conv2_aggr_modu_3NumDims, x_conv2_aggr_modu_9NumDims);

            % Add:
            x_conv2_Add_1_output = x_conv2_aggr_module_ + this.Vars.conv2_bias;
            x_conv2_Add_1_outputNumDims = max(x_conv2_aggr_module_NumDims, this.NumDims.conv2_bias);

            % Relu:
            x_Relu_1_output_0 = relu(dlarray(x_conv2_Add_1_output));
            x_Relu_1_output_0NumDims = x_conv2_Add_1_outputNumDims;

            % Shape:
            [x_conv3_Shape_output, x_conv3_Shape_outputNumDims] = model.ops.onnxShape(x_Relu_1_output_0, x_Relu_1_output_0NumDims, 0, x_Relu_1_output_0NumDims+1);

            % Slice:
            [Indices, x_conv3_Slice_outputNumDims] = model.ops.prepareSliceArgs(x_conv3_Shape_output, this.Vars.x_conv3_Constant_1_o, this.Vars.x_conv3_Constant_2_o, this.Vars.x_conv3_Constant_out, '', x_conv3_Shape_outputNumDims);
            x_conv3_Slice_output = x_conv3_Shape_output(Indices{:});

            % Squeeze:
            [x_conv3_Squeeze_outp, x_conv3_Squeeze_outpNumDims] = model.ops.onnxSqueeze(x_conv3_Slice_output, this.Vars.SqueezeAxes1019, x_conv3_Slice_outputNumDims);

            % Cast:
            x_conv3_Cast_output_ = cast(int64(extractdata(x_conv3_Squeeze_outp)), 'like', x_conv3_Squeeze_outp);
            x_conv3_Cast_output_NumDims = x_conv3_Squeeze_outpNumDims;

            % Range:
            x_conv3_Range_output = this.x_conv3_Constant_3_o:this.x_conv3_Constant_4_o:x_conv3_Cast_output_-sign(this.x_conv3_Constant_4_o)';
            x_conv3_Range_outputNumDims = 1;

            % Reshape:
            [shape, x_conv3_Reshape_outpNumDims] = model.ops.prepareReshapeArgs(x_conv3_Range_output, this.Vars.x_conv3_Constant_5_o, x_conv3_Range_outputNumDims, 0);
            x_conv3_Reshape_outp = reshape(x_conv3_Range_output, shape{:});

            % Expand:
            [shape, x_conv3_Expand_outpuNumDims] = model.ops.prepareExpandArgs(this.x_conv3_ConstantOf_3);
            x_conv3_Expand_outpu = x_conv3_Reshape_outp + zeros(shape);

            % Tile:
            [sz, x_conv3_Tile_output_NumDims] = model.ops.prepareTileArgs(this.onnx__Tile_18);
            x_conv3_Tile_output_ = repmat(x_conv3_Expand_outpu, sz);

            % Concat:
            [x_conv3_Concat_outpu, x_conv3_Concat_outpuNumDims] = model.ops.onnxConcat(1, {x_conv1_Gather_2_out, x_conv3_Tile_output_}, [x_conv1_Gather_2_outNumDims, x_conv3_Tile_output_NumDims]);

            % Shape:
            [x_conv3_Shape_1_outp, x_conv3_Shape_1_outpNumDims] = model.ops.onnxShape(x_conv3_Concat_outpu, x_conv3_Concat_outpuNumDims, 0, x_conv3_Concat_outpuNumDims+1);

            % Gather:
            [x_conv3_Gather_outpu, x_conv3_Gather_outpuNumDims] = model.ops.onnxGather(x_conv3_Shape_1_outp, this.Vars.x_conv3_Constant_7_o, 0, x_conv3_Shape_1_outpNumDims, this.NumDims.x_conv3_Constant_7_o);

            % Unsqueeze:
            [shape, x_conv3_Unsqueeze_ouNumDims] = model.ops.prepareUnsqueezeArgs(x_conv3_Gather_outpu, this.Vars.UnsqueezeAxes1020, x_conv3_Gather_outpuNumDims);
            x_conv3_Unsqueeze_ou = reshape(x_conv3_Gather_outpu, shape);

            % Concat:
            [x_conv3_Concat_1_out, x_conv3_Concat_1_outNumDims] = model.ops.onnxConcat(0, {x_conv3_Unsqueeze_ou}, [x_conv3_Unsqueeze_ouNumDims]);

            % ConstantOfShape:
            [x_conv3_ConstantOfSh, x_conv3_ConstantOfShNumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1021, x_conv3_Concat_1_out);

            % Gather:
            [x_conv3_Gather_1_out, x_conv3_Gather_1_outNumDims] = model.ops.onnxGather(x_conv3_Concat_outpu, this.Vars.x_conv1_Constant_out, 0, x_conv3_Concat_outpuNumDims, this.NumDims.x_conv1_Constant_out);

            % Gather:
            [x_conv3_Gather_2_out, x_conv3_Gather_2_outNumDims] = model.ops.onnxGather(x_conv3_Concat_outpu, this.Vars.x_conv1_Constant_1_o, 0, x_conv3_Concat_outpuNumDims, this.NumDims.x_conv1_Constant_1_o);

            % Reshape:
            [shape, x_conv3_Reshape_1_ouNumDims] = model.ops.prepareReshapeArgs(x_conv3_Gather_2_out, this.Vars.x_conv3_Constant_8_o, x_conv3_Gather_2_outNumDims, 0);
            x_conv3_Reshape_1_ou = reshape(x_conv3_Gather_2_out, shape{:});

            % Shape:
            [x_conv3_Shape_2_outp, x_conv3_Shape_2_outpNumDims] = model.ops.onnxShape(x_conv3_ConstantOfSh, x_conv3_ConstantOfShNumDims, 0, x_conv3_ConstantOfShNumDims+1);

            % Expand:
            [shape, x_conv3_Expand_1_outNumDims] = model.ops.prepareExpandArgs(x_conv3_Shape_2_outp);
            x_conv3_Expand_1_out = x_conv3_Reshape_1_ou + zeros(shape);

            % Unsqueeze:
            [shape, x_conv3_Unsqueeze_1_NumDims] = model.ops.prepareUnsqueezeArgs(x_conv3_Squeeze_outp, this.Vars.UnsqueezeAxes1022, x_conv3_Squeeze_outpNumDims);
            x_conv3_Unsqueeze_1_ = reshape(x_conv3_Squeeze_outp, shape);

            % Concat:
            [x_conv3_Concat_2_out, x_conv3_Concat_2_outNumDims] = model.ops.onnxConcat(0, {x_conv3_Unsqueeze_1_}, [x_conv3_Unsqueeze_1_NumDims]);

            % ConstantOfShape:
            [x_conv3_ConstantOf_1, x_conv3_ConstantOf_1NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1023, x_conv3_Concat_2_out);

            % Shape:
            [x_conv3_Shape_3_outp, x_conv3_Shape_3_outpNumDims] = model.ops.onnxShape(x_conv3_ConstantOf_1, x_conv3_ConstantOf_1NumDims, 0, x_conv3_ConstantOf_1NumDims+1);

            % ConstantOfShape:
            [x_conv3_ConstantOf_2, x_conv3_ConstantOf_2NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1024, x_conv3_Shape_3_outp);

            % ScatterElements:
            [x_conv3_ScatterEleme, x_conv3_ScatterElemeNumDims] = model.ops.onnxScatterElements(x_conv3_ConstantOf_2, x_conv3_Expand_1_out, x_conv3_ConstantOfSh, 0, "none", x_conv3_ConstantOf_2NumDims);

            % Add:
            x_conv3_Add_output_0 = x_conv3_ConstantOf_1 + x_conv3_ScatterEleme;
            x_conv3_Add_output_0NumDims = max(x_conv3_ConstantOf_1NumDims, x_conv3_ScatterElemeNumDims);

            % Pow:
            x_conv3_Pow_output_0 = power(x_conv3_Add_output_0, this.x_conv3_Constant_9_o);
            x_conv3_Pow_output_0NumDims = max(x_conv3_Add_output_0NumDims, this.NumDims.x_conv3_Constant_9_o);

            % Equal:
            x_conv3_Equal_output = x_conv3_Pow_output_0 == this.Vars.x_conv3_Constant_10_;
            x_conv3_Equal_outputNumDims = max(x_conv3_Pow_output_0NumDims, this.NumDims.x_conv3_Constant_10_);

            % Cast:
            x_conv3_Cast_1_outpu = logical(x_conv3_Equal_output);
            x_conv3_Cast_1_outpuNumDims = x_conv3_Equal_outputNumDims;

            % Where:
            [x_conv3_Where_output, x_conv3_Where_outputNumDims] = model.ops.onnxWhere(x_conv3_Cast_1_outpu, this.x_conv3_Constant_11_, x_conv3_Pow_output_0, x_conv3_Cast_1_outpuNumDims, this.NumDims.x_conv3_Constant_11_, x_conv3_Pow_output_0NumDims);

            % Gather:
            [x_conv3_Gather_3_out, x_conv3_Gather_3_outNumDims] = model.ops.onnxGather(x_conv3_Where_output, x_conv3_Gather_1_out, 0, x_conv3_Where_outputNumDims, x_conv3_Gather_1_outNumDims);

            % Mul:
            x_conv3_Mul_output_0 = x_conv3_Gather_3_out .* x_conv3_ConstantOfSh;
            x_conv3_Mul_output_0NumDims = max(x_conv3_Gather_3_outNumDims, x_conv3_ConstantOfShNumDims);

            % Gather:
            [x_conv3_Gather_4_out, x_conv3_Gather_4_outNumDims] = model.ops.onnxGather(x_conv3_Where_output, x_conv3_Gather_2_out, 0, x_conv3_Where_outputNumDims, x_conv3_Gather_2_outNumDims);

            % Mul:
            x_conv3_Mul_1_output = x_conv3_Mul_output_0 .* x_conv3_Gather_4_out;
            x_conv3_Mul_1_outputNumDims = max(x_conv3_Mul_output_0NumDims, x_conv3_Gather_4_outNumDims);

            % MatMul:
            [x_conv3_lin_MatMul_o, x_conv3_lin_MatMul_oNumDims] = model.ops.onnxMatMul(x_Relu_1_output_0, this.onnx__MatMul_273, x_Relu_1_output_0NumDims, this.NumDims.onnx__MatMul_273);

            % Shape:
            [x_conv3_Shape_4_outp, x_conv3_Shape_4_outpNumDims] = model.ops.onnxShape(x_conv3_lin_MatMul_o, x_conv3_lin_MatMul_oNumDims, 0, x_conv3_lin_MatMul_oNumDims+1);

            % Slice:
            [Indices, x_conv3_Slice_1_outpNumDims] = model.ops.prepareSliceArgs(x_conv3_Shape_4_outp, this.Vars.x_conv3_Constant_13_, this.Vars.x_conv3_Constant_14_, this.Vars.x_conv3_Constant_12_, '', x_conv3_Shape_4_outpNumDims);
            x_conv3_Slice_1_outp = x_conv3_Shape_4_outp(Indices{:});

            % Squeeze:
            [x_conv3_Squeeze_1_ou, x_conv3_Squeeze_1_ouNumDims] = model.ops.onnxSqueeze(x_conv3_Slice_1_outp, this.Vars.SqueezeAxes1025, x_conv3_Slice_1_outpNumDims);

            % Gather:
            [x_conv3_Gather_5_out, x_conv3_Gather_5_outNumDims] = model.ops.onnxGather(x_conv3_lin_MatMul_o, x_conv3_Gather_1_out, -2, x_conv3_lin_MatMul_oNumDims, x_conv3_Gather_1_outNumDims);

            % Reshape:
            [shape, x_conv3_Reshape_2_ouNumDims] = model.ops.prepareReshapeArgs(x_conv3_Mul_1_output, this.Vars.x_conv3_Constant_15_, x_conv3_Mul_1_outputNumDims, 0);
            x_conv3_Reshape_2_ou = reshape(x_conv3_Mul_1_output, shape{:});

            % Mul:
            x_conv3_Mul_2_output = x_conv3_Reshape_2_ou .* x_conv3_Gather_5_out;
            x_conv3_Mul_2_outputNumDims = max(x_conv3_Reshape_2_ouNumDims, x_conv3_Gather_5_outNumDims);

            % Shape:
            [x_conv3_aggr_modu_12, x_conv3_aggr_modu_12NumDims] = model.ops.onnxShape(x_conv3_Mul_2_output, x_conv3_Mul_2_outputNumDims, 0, x_conv3_Mul_2_outputNumDims+1);

            % Gather:
            [x_conv3_aggr_modu_7, x_conv3_aggr_modu_7NumDims] = model.ops.onnxGather(x_conv3_aggr_modu_12, this.Vars.x_conv3_aggr_modu_5, 0, x_conv3_aggr_modu_12NumDims, this.NumDims.x_conv3_aggr_modu_5);

            % Reshape:
            [shape, x_conv3_aggr_modu_8NumDims] = model.ops.prepareReshapeArgs(x_conv3_Gather_2_out, this.Vars.x_conv3_aggr_modu_4, x_conv3_Gather_2_outNumDims, 0);
            x_conv3_aggr_modu_8 = reshape(x_conv3_Gather_2_out, shape{:});

            % Shape:
            [x_conv3_aggr_modu_10, x_conv3_aggr_modu_10NumDims] = model.ops.onnxShape(x_conv3_Mul_2_output, x_conv3_Mul_2_outputNumDims, 0, x_conv3_Mul_2_outputNumDims+1);

            % Expand:
            [shape, x_conv3_aggr_modu_6NumDims] = model.ops.prepareExpandArgs(x_conv3_aggr_modu_10);
            x_conv3_aggr_modu_6 = x_conv3_aggr_modu_8 + zeros(shape);

            % Unsqueeze:
            [shape, x_conv3_aggr_modu_14NumDims] = model.ops.prepareUnsqueezeArgs(x_conv3_Squeeze_1_ou, this.Vars.UnsqueezeAxes1026, x_conv3_Squeeze_1_ouNumDims);
            x_conv3_aggr_modu_14 = reshape(x_conv3_Squeeze_1_ou, shape);

            % Unsqueeze:
            [shape, x_conv3_aggr_modu_13NumDims] = model.ops.prepareUnsqueezeArgs(x_conv3_aggr_modu_7, this.Vars.UnsqueezeAxes1027, x_conv3_aggr_modu_7NumDims);
            x_conv3_aggr_modu_13 = reshape(x_conv3_aggr_modu_7, shape);

            % Concat:
            [x_conv3_aggr_modu_1, x_conv3_aggr_modu_1NumDims] = model.ops.onnxConcat(0, {x_conv3_aggr_modu_14, x_conv3_aggr_modu_13}, [x_conv3_aggr_modu_14NumDims, x_conv3_aggr_modu_13NumDims]);

            % ConstantOfShape:
            [x_conv3_aggr_modu_3, x_conv3_aggr_modu_3NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1028, x_conv3_aggr_modu_1);

            % Shape:
            [x_conv3_aggr_modu_11, x_conv3_aggr_modu_11NumDims] = model.ops.onnxShape(x_conv3_aggr_modu_3, x_conv3_aggr_modu_3NumDims, 0, x_conv3_aggr_modu_3NumDims+1);

            % ConstantOfShape:
            [x_conv3_aggr_modu_2, x_conv3_aggr_modu_2NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1029, x_conv3_aggr_modu_11);

            % ScatterElements:
            [x_conv3_aggr_modu_9, x_conv3_aggr_modu_9NumDims] = model.ops.onnxScatterElements(x_conv3_aggr_modu_2, x_conv3_aggr_modu_6, x_conv3_Mul_2_output, 0, "none", x_conv3_aggr_modu_2NumDims);

            % Add:
            x_conv3_aggr_module_ = x_conv3_aggr_modu_3 + x_conv3_aggr_modu_9;
            x_conv3_aggr_module_NumDims = max(x_conv3_aggr_modu_3NumDims, x_conv3_aggr_modu_9NumDims);

            % Add:
            x_conv3_Add_1_output = x_conv3_aggr_module_ + this.Vars.conv3_bias;
            x_conv3_Add_1_outputNumDims = max(x_conv3_aggr_module_NumDims, this.NumDims.conv3_bias);

            % Shape:
            [x_Shape_output_0, x_Shape_output_0NumDims] = model.ops.onnxShape(x_conv3_Add_1_output, x_conv3_Add_1_outputNumDims, 0, x_conv3_Add_1_outputNumDims+1);

            % Gather:
            [x_Gather_output_0, x_Gather_output_0NumDims] = model.ops.onnxGather(x_Shape_output_0, this.Vars.x_Constant_output_0, 0, x_Shape_output_0NumDims, this.NumDims.x_Constant_output_0);

            % Shape:
            [x_Shape_1_output_0, x_Shape_1_output_0NumDims] = model.ops.onnxShape(x_conv3_Add_1_output, x_conv3_Add_1_outputNumDims, 0, x_conv3_Add_1_outputNumDims+1);

            % Gather:
            [x_Gather_1_output_0, x_Gather_1_output_0NumDims] = model.ops.onnxGather(x_Shape_1_output_0, this.Vars.x_Constant_1_output_, 0, x_Shape_1_output_0NumDims, this.NumDims.x_Constant_1_output_);

            % Unsqueeze:
            [shape, x_Unsqueeze_output_0NumDims] = model.ops.prepareUnsqueezeArgs(x_Gather_1_output_0, this.Vars.UnsqueezeAxes1030, x_Gather_1_output_0NumDims);
            x_Unsqueeze_output_0 = reshape(x_Gather_1_output_0, shape);

            % Concat:
            [x_Concat_output_0, x_Concat_output_0NumDims] = model.ops.onnxConcat(0, {x_Unsqueeze_output_0}, [x_Unsqueeze_output_0NumDims]);

            % ConstantOfShape:
            [x_ConstantOfShape_ou, x_ConstantOfShape_ouNumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1031, x_Concat_output_0);

            % ScatterElements:
            [x_ScatterElements_ou, x_ScatterElements_ouNumDims] = model.ops.onnxScatterElements(this.x_Constant_2_output_, batch, x_ConstantOfShape_ou, 0, "none", this.NumDims.x_Constant_2_output_);

            % Add:
            x_Add_output_0 = this.Vars.x_Constant_3_output_ + x_ScatterElements_ou;
            x_Add_output_0NumDims = max(this.NumDims.x_Constant_3_output_, x_ScatterElements_ouNumDims);

            % Clip:
            x_Clip_output_0 = min(Inf, max(this.Vars.x_Constant_4_output_, x_Add_output_0));
            x_Clip_output_0NumDims = x_Add_output_0NumDims;

            % Reshape:
            [shape, x_Reshape_output_0NumDims] = model.ops.prepareReshapeArgs(batch, this.Vars.x_Constant_5_output_, batchNumDims, 0);
            x_Reshape_output_0 = reshape(batch, shape{:});

            % Shape:
            [x_Shape_2_output_0, x_Shape_2_output_0NumDims] = model.ops.onnxShape(x_conv3_Add_1_output, x_conv3_Add_1_outputNumDims, 0, x_conv3_Add_1_outputNumDims+1);

            % Expand:
            [shape, x_Expand_output_0NumDims] = model.ops.prepareExpandArgs(x_Shape_2_output_0);
            x_Expand_output_0 = x_Reshape_output_0 + zeros(shape);

            % Unsqueeze:
            [shape, x_Unsqueeze_1_outputNumDims] = model.ops.prepareUnsqueezeArgs(x_Gather_output_0, this.Vars.UnsqueezeAxes1032, x_Gather_output_0NumDims);
            x_Unsqueeze_1_output = reshape(x_Gather_output_0, shape);

            % Concat:
            [x_Concat_1_output_0, x_Concat_1_output_0NumDims] = model.ops.onnxConcat(0, {this.Vars.x_Constant_6_output_, x_Unsqueeze_1_output}, [this.NumDims.x_Constant_6_output_, x_Unsqueeze_1_outputNumDims]);

            % ConstantOfShape:
            [x_ConstantOfShape_1_, x_ConstantOfShape_1_NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1033, x_Concat_1_output_0);

            % Shape:
            [x_Shape_3_output_0, x_Shape_3_output_0NumDims] = model.ops.onnxShape(x_ConstantOfShape_1_, x_ConstantOfShape_1_NumDims, 0, x_ConstantOfShape_1_NumDims+1);

            % ConstantOfShape:
            [x_ConstantOfShape_2_, x_ConstantOfShape_2_NumDims] = model.ops.onnxConstantOfShape(this.Vars.ConstantOfShapeValue1034, x_Shape_3_output_0);

            % ScatterElements:
            [x_ScatterElements_1_, x_ScatterElements_1_NumDims] = model.ops.onnxScatterElements(x_ConstantOfShape_2_, x_Expand_output_0, x_conv3_Add_1_output, 0, "none", x_ConstantOfShape_2_NumDims);

            % Add:
            x_Add_1_output_0 = x_ConstantOfShape_1_ + x_ScatterElements_1_;
            x_Add_1_output_0NumDims = max(x_ConstantOfShape_1_NumDims, x_ScatterElements_1_NumDims);

            % Reshape:
            [shape, x_Reshape_1_output_0NumDims] = model.ops.prepareReshapeArgs(x_Clip_output_0, this.Vars.x_Constant_7_output_, x_Clip_output_0NumDims, 0);
            x_Reshape_1_output_0 = reshape(x_Clip_output_0, shape{:});

            % Shape:
            [x_Shape_4_output_0, x_Shape_4_output_0NumDims] = model.ops.onnxShape(x_Add_1_output_0, x_Add_1_output_0NumDims, 0, x_Add_1_output_0NumDims+1);

            % Expand:
            [shape, x_Expand_1_output_0NumDims] = model.ops.prepareExpandArgs(x_Shape_4_output_0);
            x_Expand_1_output_0 = x_Reshape_1_output_0 + zeros(shape);

            % Div:
            x_Div_output_0 = x_Add_1_output_0 ./ x_Expand_1_output_0;
            x_Div_output_0NumDims = max(x_Add_1_output_0NumDims, x_Expand_1_output_0NumDims);

            % Gemm:
            [A, B, C, alpha, beta, outputNumDims] = model.ops.prepareGemmArgs(x_Div_output_0, this.lin_weight, this.lin_bias, this.Vars.Gemmalpha1035, this.Vars.Gemmbeta1036, 0, 1, this.NumDims.lin_bias);
            output = alpha*B*A + beta*C;

            % Set graph output arguments
            outputNumDims1037 = outputNumDims;

        end

    end

end