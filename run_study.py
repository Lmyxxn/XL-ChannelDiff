import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_root', default='study')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr_g', type=float, default=0.0002)
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--loss_type', choices=('l1', 'l2'), default='l1')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = output / 'status.json'
    def write_status(state, **extra):
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=state, pid=os.getpid(), **extra), indent=2))
        os.replace(temporary, status)
    try:
        write_status('training', seed=42, requested_epochs=args.epochs,
                     lr_g=args.lr_g, lr_d=args.lr_d, warmup_steps=args.warmup_steps,
                     loss_type=args.loss_type, gpu=args.gpu)
        subprocess.run([sys.executable, '-u', 'train.py', '--gpu', args.gpu, '--seed', '42',
                        '--epochs', str(args.epochs), '--data_root', args.data_root,
                        '--lr_g', str(args.lr_g), '--lr_d', str(args.lr_d),
                        '--warmup_steps', str(args.warmup_steps),
                        '--loss_type', args.loss_type,
                        '--resume', args.resume,
                        '--output_root', str(output / 'training')], cwd=root, check=True)
        files = list((output / 'training').rglob('model_best.pt'))
        if len(files) != 1:
            raise RuntimeError(f'Expected one best checkpoint, found {len(files)}.')
        checkpoint = files[0]
        write_status('evaluating', checkpoint=str(checkpoint))
        for steps in (5, 50, 100):
            subprocess.run([sys.executable, '-u', 'evaluate.py', '--checkpoint', str(checkpoint),
                            '--data_root', args.data_root, '--ddim_steps', str(steps),
                            '--output_root', str(output / f'evaluation_{steps}steps')],
                           cwd=root, check=True)
        write_status('completed', checkpoint=str(checkpoint))
    except Exception as error:
        write_status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
