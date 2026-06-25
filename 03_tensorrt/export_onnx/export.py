import torch
import torch.nn as nn


ONNX_PATH = "03_tensorrt/export_onnx/model.onnx"


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


def main():
    torch.manual_seed(0)

    model = TinyModel().eval()
    x = torch.randn(1, 16)

    with torch.no_grad():
        y = model(x)

    torch.onnx.export(
        model,
        x,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("onnx path:", ONNX_PATH)

    try:
        import onnx

        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print("onnx check: passed")
    except ImportError:
        print("onnx check: skipped because package 'onnx' is not installed")


if __name__ == "__main__":
    main()
