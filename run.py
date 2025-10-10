import torch
import argparse
import train, rec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=int, default=0, help='Run mode')
    parser.add_argument('--model_id', type=int, default=5, help='Model ID')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--scheduler_step', type=int, default=20, help='Scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Scheduler gamma')
    parser.add_argument('--nepoch', nargs='+', type=int, default=[2], help='Number of epochs (Single value for combined training; Array for one-by-one training)')
    parser.add_argument('--nevt', type=int, default=100, help='Number of events')
    parser.add_argument('--nevt_val', type=int, default=10, help='Number of events for validation')
    parser.add_argument('--tag', type=str, default='cnn', help='Name tag')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--save_checkpoints', type=int, default=None, help='Save checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Resume checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--dataset_train', type=str, default='../samples/dataset_2k_noise100_ztrk.root', help='Training dataset')
    parser.add_argument('--dataset_test', type=str, default='../samples/dataset_2k_noise100_ztrk.root', help='Testing dataset')
    parser.add_argument('--dataset_rec', type=str, default='../samples/dataset_test.root', help='Reconstruction dataset')
    parser.add_argument('--prob_cut', type=float, default=0.5, help='Probability cut')
    parser.add_argument('--model_layers', nargs='+', type=int, default=(32, 128, 512, 2048, 8192), help='Number of layers')
    parser.add_argument('--model_down_ratio', type=float, default=0.25, help='Down ratio')
    parser.add_argument('--enable_debug', action='store_true', help='Enable debug')
    parser.add_argument('--knn', type=int, default=16, help='Number of neighbors')
    parser.add_argument('--undirected', action='store_true', help='Undirected graph')
    parser.add_argument('--reduce', type=str, default='max', help='Reduce method')
    parser.add_argument('--outlier_distance', type=float, default=-1, help='Outlier distance')
    parser.add_argument('--enable_balanced_loss', action='store_true', help='Enable balanced loss')
    parser.add_argument('--enable_self_attention', action='store_true', help='Enable self attention')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--use_truth_position', action='store_true', help='Use truth position')
    parser.add_argument('--pos_off', action='store_true', help='Position off')

    parser.add_argument('--model', type=str, default='./results/model.pth')
    parser.add_argument('--enable_trad_rec', action='store_true', help='Enable traditional reconstruction')
    parser.add_argument('--dndx_trunc_layers', type=int, default=2, help='Number of layers for dndx truncation')
    parser.add_argument('--dndx_trunc_ratio', type=float, default=0.65, help='Ratio for dndx truncation')
    parser.add_argument('--dedx_trunc_layers', type=int, default=1, help='Number of layers for dedx truncation')
    parser.add_argument('--dedx_trunc_ratio', type=float, default=0.6, help='Ratio for dedx truncation')
    parser.add_argument('--rec_with_checkpoint', action='store_true', help='Reconstruction with checkpoint')
    args = parser.parse_args()

    print("Parsed Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    if args.run_mode == 0:
        train.main(args)
    elif args.run_mode == 1:
        rec.main(args)