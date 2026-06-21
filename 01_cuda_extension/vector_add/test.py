import torch
import vector_add_ext


def main():
    a = torch.ones(8, device="cuda", dtype=torch.float32)
    b = torch.ones(8, device="cuda", dtype=torch.float32)

    c = vector_add_ext.vector_add(a, b)

    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("expected:", a + b)
    print("allclose:", torch.allclose(c, a + b))


if __name__ == "__main__":
    main()