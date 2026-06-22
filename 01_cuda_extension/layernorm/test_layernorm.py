import torch
import layernorm_ext


def main():
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    y_ext = layernorm_ext.layernorm(x)
    y_ref = torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[-1],))

    print("x:")
    print(x)

    print("y_ext:")
    print(y_ext)

    print("y_ref:")
    print(y_ref)

    print("max error:", (y_ext - y_ref).abs().max().item())
    print("allclose:", torch.allclose(y_ext, y_ref, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    main()