
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='HCPN')

    parser.add_argument('-year', dest='year', default='2017')
    parser.add_argument('-imsize', dest='imsize', default=480, type=int)
    parser.add_argument('-batch_size', dest='batch_size', default=4, type=int)
    parser.add_argument('-num_workers', dest='num_workers', default=8, type=int)

    ## TRAINING parameters ##
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help=('whether to resume training an existing model'
                              '(the one with name model_name will be used)'))
    parser.set_defaults(resume=False)
    parser.add_argument('-seed', dest='seed', default=123, type=int)
    # parser.add_argument('-gpu_id', dest='gpu_id', default=0, type=int)
    parser.add_argument('-lr', dest='lr', default=1e-3, type=float) #decoder
    parser.add_argument('-lr_cnn', dest='lr_cnn', default=1e-3, type=float) #encoder
    parser.add_argument('-lr_stepsize', dest='lr_stepsize', default=2)
    parser.add_argument('-lr_gamma', dest='lr_gamma', default=0.5)
    parser.add_argument('-optim_cnn', dest='optim_cnn', default='sgd',
                        choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('-momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('-weight_decay', dest='weight_decay', default=5e-4,
                        type=float)
    parser.add_argument('-weight_decay_cnn', dest='weight_decay_cnn',
                        default=5e-4, type=float)
    parser.add_argument('-optim', dest='optim', default='sgd',
                        choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.set_defaults(crop=False)

    parser.add_argument('--update_encoder', dest='update_encoder',
                        action='store_true',
                        help='used in sync with finetune_after.'
                             ' no need to activate.')
    parser.set_defaults(update_encoder=True)

    parser.add_argument('-max_epoch', dest='max_epoch', default=30, type=int)

    # visualization and logging
    parser.add_argument('-print_every', dest='print_every', default=10,
                        type=int)
    
    # loss weights
    parser.add_argument('-iou_weight', dest='iou_weight', default=1.0,
                        type=float)
    # augmentation
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.set_defaults(augment=False)
    parser.add_argument('-rotation', dest='rotation', default=10, type=int)
    parser.add_argument('-translation', dest='translation', default=0.1,
                        type=float)
    parser.add_argument('-shear', dest='shear', default=0.1, type=float)
    parser.add_argument('-zoom', dest='zoom', default=0.7, type=float)

    # GPU
    parser.add_argument('--cpu', dest='use_gpu', action='store_false')
    parser.set_defaults(use_gpu=True)
    parser.add_argument('-ngpus', dest='ngpus', default=1, type=int)

    parser.add_argument('-model_name', dest='model_name', default='HCPN_3rgb')
    #result,存储日志文件和权重
    parser.add_argument('-result_path', dest='result_path', default='/users/u202220081200014/video-inpainting/video_forgery/SCFNet_2.0/results')

    # dataset parameters
    parser.add_argument('--resize', dest='resize', action='store_true')
    parser.set_defaults(resize=False)
    parser.add_argument('-dataset', dest='dataset', default='youtube',
                        choices=['davis2017', 'youtube'])

    # testing
    parser.add_argument('-eval_split', dest='eval_split', default='test')
    parser.add_argument('-mask_th', dest='mask_th', default=0.5, type=float)
    parser.add_argument('-max_dets', dest='max_dets', default=100, type=int)
    parser.add_argument('-min_size', dest='min_size', default=0.001,
                        type=float)
    parser.add_argument('--display', dest='display', action='store_true')
    parser.add_argument('--no_display_text', dest='no_display_text',
                        action='store_true')
    parser.set_defaults(display=False)
    parser.set_defaults(display_route=False)
    parser.set_defaults(no_display_text=False)
    parser.set_defaults(use_gt_masks=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()

