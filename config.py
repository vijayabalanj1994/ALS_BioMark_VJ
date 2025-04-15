import torch

class Config():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    device = torch.device(device)

    seed = 9
    classification = "multi-classification" # can be binary-classification or multi-classification

    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    epoch = 300
    batch_size =64
    lr = 0.0001
    weight_decay = 0.001

    # data internal
    img_w, img_h = 400, 400
    train_loader = None
    val_loader = None
    test_loader = None

    # model internal
    model = None
    criterion = None
    optimizer = None
    scheduler = None

config = Config()