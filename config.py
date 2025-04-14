class Config():

    # path to the dataset
    main_path = "./"
    dataset_path = "dataset/AptamerROIs020623"
    #label_path = "dataset/RoI Image Key.xlsx"

    # model config
    classification = "multi-classification" # can be binary-classification or multi-classification
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    epoch = 150
    batch_size =150
    lr = 0.0001
    seed = 9
    model = "CNN"

config = Config()