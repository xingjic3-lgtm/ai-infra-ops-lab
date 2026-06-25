from pathlib import Path

import tensorrt as trt


ONNX_PATH = Path("03_tensorrt/export_onnx/model.onnx")
ENGINE_PATH = Path("03_tensorrt/build_engine/model.engine")


def main():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if not parser.parse_from_file(str(ONNX_PATH)):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError("failed to parse ONNX model")

    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, (1, 16), (4, 16), (16, 16))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("failed to build TensorRT engine")

    ENGINE_PATH.write_bytes(engine_bytes)
    print(f"saved engine to {ENGINE_PATH}")


if __name__ == "__main__":
    main()
