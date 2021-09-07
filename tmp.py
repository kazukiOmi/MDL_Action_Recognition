import torch


def pre_model():
    # model = torch.hub.load('facebookresearch/pytorchvideo',
    #                        'x3d_xs', pretrained=True)
    # model = torch.hub.load(
    #     'facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
    model_x3d_m = torch.hub.load(
        'facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
    model_slow_r50 = torch.hub.load('facebookresearch/pytorchvideo',
                                    'slow_r50', pretrained=True)


if __name__ == '__main__':
    pre_model()
    import dataset_reader
    dataset_reader.main()
