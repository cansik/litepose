import onnx
import onnxoptimizer

src_onnx = "models/litepose-auto-m-coco.onnx"
opt_onnx = "models/litepose-auto-m-coco-opt.onnx"

model = onnx.load(src_onnx)

# optimize
onnx.checker.check_model(model)
# inferred_model = onnx.shape_inference.infer_shapes(model)
# model = onnx.optimizer.optimize(inferred_model, passes=['fuse_bn_into_conv'])

for init in model.graph.initializer:
    for value_info in model.graph.value_info:
        if init.name == value_info.name:
            model.graph.input.append(value_info)

model = onnxoptimizer.optimize(model, ['fuse_bn_into_conv'])

# save optimized model
with open(opt_onnx, "wb") as f:
    f.write(model.SerializeToString())
print("done")
