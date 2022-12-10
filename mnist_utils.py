from torchvision.utils import save_image
from GAN_client.core.utils import scale_image



def save_images_mnist():
    from torchvision.datasets import CIFAR10, MNIST
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,),
                             std=(0.5,))])

    test_dataset = MNIST(
        root='.',
        train=False,
        transform=transform,
        download=True)
    data_loader = DataLoader(dataset=test_dataset,
                                                batch_size=1,
                                                shuffle=False)
    counter = 1
    for inputs, _ in data_loader:
        # don't need targets
        n = inputs.size(0)
        
        for j in range(1,n+1):        
            save_image(scale_image(inputs[j-1]), f"MNIST/images/{counter}.png")
            counter += 1


if __name__ == "__main__":
    save_images_mnist()