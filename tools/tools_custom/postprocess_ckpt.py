import os
import argparse
from collections import OrderedDict
import torch


def get_arg():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_prefix', type=str, default="pretrained")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arg()
    dir_name, model_name = os.path.dirname(args.model_path), os.path.basename(args.model_path)
    out_path = os.path.join(dir_name, args.model_prefix+'_'+model_name)

    ckpt = torch.load(args.model_path)
    print(ckpt.keys())
    assert 'state_dict' in ckpt.keys()

    # post-processing on pre-trained model weights
    # 1. remove mismatched layers in state_dict (reference_points, cls_branches, and query_embedding)
    # 2. rename the norm layers in encoder2 layers (with +1 index)
    #    the attn layers in encoder2 are not rename for shared prompt encoder
    state_dict_new = OrderedDict()
    for key, v in ckpt['state_dict'].items():
        key_split = key.split('.')
        if ('bbox_head.transformer.reference_points' in key
                or 'bbox_head.cls_branches' in key
                or 'bbox_head.query_embedding' in key):
            print('removed: ', key, '\n')
            continue
        elif key.startswith('bbox_head.transformer.encoder2') and key_split[5] == 'norms':
            key_split[6] = str(int(key_split[6]) + 1)
            new_key = '.'.join(key_split)
            print(key, '  ---->  ')
            print(new_key, '\n')
        else:
            new_key = key
            print('copied: ', key, '\n')
        state_dict_new[new_key] = v

    ckpt_new = {'state_dict': state_dict_new}
    torch.save(ckpt_new, out_path)