import torch
from model import TheWorld

def main():
    model = TheWorld("google/gemma-3-1b-it")
    model.forward(torch.randn(1, 3, 224, 224), torch.randint(0, 100, (1, 10)), torch.randint(0, 1, (1, 10)))


if __name__ == "__main__":
    main()
