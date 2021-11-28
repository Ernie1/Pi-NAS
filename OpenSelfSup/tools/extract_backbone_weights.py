import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    parser.add_argument(
        '--part', type=str, default='backbone', help='part of weights'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author="OpenSelfSup")
    has_backbone = False
    for key, value in ck['state_dict'].items():
        if key.startswith(args.part):
            output_dict['state_dict'][key[len(args.part) + 1:]] = value
            has_backbone = True
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
