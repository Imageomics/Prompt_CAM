import torch
from data.dataset.cub import get_cub
from data.dataset.dog import get_dog
from data.dataset.pet import get_pet
from data.dataset.car import get_car
from data.dataset.birds_525 import get_birds_525


def get_dataset(data, params, logger):
    dataset_train, dataset_val, dataset_test = None, None, None

    if data.startswith("cub"):
        logger.info("Loading CUB data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for cub)...")
            dataset_train = get_cub(params, 'trainval_combined')
            dataset_test = get_cub(params, 'test')
        else:
            raise NotImplementedError 
    elif data.startswith("dog"):
        logger.info("Loading Standford Dogs data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for dog)...")
            dataset_train = get_dog(params, 'trainval_combined')
            dataset_test = get_dog(params, 'test')
        else:
            raise NotImplementedError
    elif data.startswith("pet"):
        logger.info("Loading Oxford Pet data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for pet)...")
            dataset_train = get_pet(params, 'trainval_combined')
            dataset_test = get_pet(params, 'test')
        else:
            raise NotImplementedError
    elif data.startswith("car"):
        logger.info("Loading Stanford Car data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for car)...")
            dataset_train = get_car(params, 'trainval_combined')
            dataset_test = get_car(params, 'test')
        else:
            raise NotImplementedError
    elif data.startswith("birds_525"):
        logger.info("Loading Birds 525 data ...")
        if params.final_run:
            logger.info("Loading training data (final training data for birds_525)...")
            dataset_train = get_birds_525(params, 'trainval_combined')
            dataset_test = get_birds_525(params, 'test')
        else:
            raise NotImplementedError
    else:
        raise Exception("Dataset '{}' not supported".format(data))
    return dataset_train, dataset_val, dataset_test


def get_loader(params, logger):
    if 'test_data' in params:
        dataset_train, dataset_val, dataset_test = get_dataset(params.test_data, params, logger)
    else:
        dataset_train, dataset_val, dataset_test = get_dataset(params.data, params, logger)

    if isinstance(dataset_train, list):
        train_loader, val_loader, test_loader = [], [], []
        for i in range(len(dataset_train)):
            tmp_train, tmp_val, tmp_test = gen_loader(params, dataset_train[i], dataset_val[i], None)
            train_loader.append(tmp_train)
            val_loader.append(tmp_val)
            test_loader.append(tmp_test)
    else:
        train_loader, val_loader, test_loader = gen_loader(params, dataset_train, dataset_val, dataset_test)

    logger.info("Finish setup loaders")
    return train_loader, val_loader, test_loader


def gen_loader(params, dataset_train, dataset_val, dataset_test):
    train_loader, val_loader, test_loader = None, None, None
    if params.debug:
        num_workers = 1
    else:
        num_workers = 4
    if dataset_train is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    if dataset_val is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    if dataset_test is not None:
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True

        )
    return train_loader, val_loader, test_loader
