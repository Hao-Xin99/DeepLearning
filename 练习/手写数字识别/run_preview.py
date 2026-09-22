import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from minist_can import Net, get_data_loader, device, evaluate

def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    net.to(device)

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(3):
        for (x, y) in train_data:
            net.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    net.eval()
    with torch.no_grad():
        plt.figure(figsize=(15, 3))
        for i in range(10):
            image, label = test_data.dataset[i]
            image_gpu = image.to(device)
            predict = torch.argmax(net(image_gpu.unsqueeze(0)))
            plt.subplot(1, 10, i + 1)
            plt.imshow(image.view(28, 28))
            color = "green" if int(predict) == int(label) else "red"
            plt.title(f"pred:{int(predict)}\ntrue:{int(label)}", color=color)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("prediction_result.png", dpi=100, bbox_inches="tight")
        print("图片已保存: prediction_result.png")

if __name__ == "__main__":
    main()
