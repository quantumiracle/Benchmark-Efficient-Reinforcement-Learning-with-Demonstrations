import argparse
import numpy as np

from maml import MAML
''' train '''
# python main.py --dataset sin --K 5 --num_updates 1 --norm None --is_train --max_steps 10000
''' evaluate '''
# python main.py --dataset sin --K 5 --num_updates 5 --norm None --restore_checkpoint './checkpoint/MAML.sin_5-shot_1-updates_25-batch_norm-None/checkpoint-9000' --draw
def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of MAML")
    parser.add_argument('--seed', type=int, default=1)
    # Dataset
    parser.add_argument('--dataset', help='environment ID', choices=['sin'],
                        required=True)
    # MAML
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='fc')
    parser.add_argument('--loss_type', type=str, default='MSE')
    parser.add_argument('--num_updates', type=int, default=3)
    parser.add_argument('--norm', choices=['None', 'batch_norm'], default='batch_norm')
    # Train
    parser.add_argument('--is_train', action='store_true', default=False)
    parser.add_argument('--max_steps', type=int, default=7e4)
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=25)
    # Test
    parser.add_argument('--restore_checkpoint', type=str)
    parser.add_argument('--restore_dir', type=str)
    parser.add_argument('--test_sample', type=int, default=100)
    parser.add_argument('--draw', action='store_true', default=False)
    args = parser.parse_args()
    return args


def get_dataset(dataset_name, K_shots):
    if dataset_name == 'sin':
        from dataset.SinDataGenerator import dataset
    else:
        ValueError("Invalid dataset")
    return dataset(K_shots)


def main(args):
    np.random.seed(args.seed)
    dataset = get_dataset(args.dataset, args.K)
    model = MAML(dataset,
                 args.model_type,
                 args.loss_type,
                 dataset.dim_input,
                 dataset.dim_output,
                 args.alpha,
                 args.beta,
                 args.K,
                 args.batch_size,
                 args.is_train,
                 args.num_updates,
                 args.norm
                 )
    if args.is_train:
        model.learn(args.batch_size, dataset, args.max_steps)
    else:
        model.evaluate(dataset, args.test_sample, args.draw,
                       restore_checkpoint=args.restore_checkpoint,
                       restore_dir=args.restore_dir)

if __name__ == '__main__':
    args = argsparser()
    main(args)
