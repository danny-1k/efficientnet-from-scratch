if __name__ == '__main__':

    from torchvision import transforms

    import argparse
    from models import EfficientNet
    
    from utils import read_image

    import json

    parser = argparse.ArgumentParser()

    parser.add_argument('--image', default='doggo.jpg')
    parser.add_argument('--model', default='b0')

    args = parser.parse_args()

    net = EfficientNet(args.model)
    net.from_pretrained(args.model)
    net.eval()
    net.requires_grad_(False)

    #imagenet transforms

    transform = transforms.Compose([
        transforms.Resize((net.image_size, net.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    x = read_image(args.image)
    x = transform(x)

    x = x.unsqueeze(0)

    prediction = net(x)

    prediction_index = str(prediction.argmax(axis=1).item())
    
    labels = json.load(open('labels.txt', 'rb'))
    label = labels[prediction_index] 


    print(f'I think the image is of a(an) -> {label}')

