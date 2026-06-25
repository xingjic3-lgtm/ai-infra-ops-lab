from pathlib import Path

import tensorrt as trt


ENGINE_PATH = Path("03_tensorrt/build_engine/model.engine")


def main():
    if not ENGINE_PATH.exists():
        raise FileNotFoundError(f"missing engine file: {ENGINE_PATH}")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    engine_bytes = ENGINE_PATH.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("failed to deserialize TensorRT engine")

    print(f"engine path: {ENGINE_PATH}")
    print(f"engine size: {ENGINE_PATH.stat().st_size} bytes")
    print(f"num io tensors: {engine.num_io_tensors}")

    print("io tensors:")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        print(f"  {index}: name={name}, mode={mode}, dtype={dtype}, shape={tuple(shape)}")

    inspector = engine.create_engine_inspector()
    print("engine inspector:")
    print(inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))


if __name__ == "__main__":
    main()