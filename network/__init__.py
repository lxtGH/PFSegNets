"""
Network Initializations
"""

import logging
import importlib
import torch


def get_net(args, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(network=args.arch, num_classes=args.dataset_cls.num_classes,
                    criterion=criterion, args=args)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def wrap_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        import apex
        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network, num_classes, criterion, args):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    if model == 'DeepR101_PF_maxavg_deeply' or model == 'DeepR50_PF_maxavg_deeply':
        net = net_func(num_classes=num_classes, criterion=criterion, reduce_dim=args.match_dim,
                       max_pool_size=args.maxpool_size, avgpool_size=args.avgpool_size, edge_points=args.edge_points)
    else:
        net = net_func(num_classes=num_classes, criterion=criterion)
    return net
