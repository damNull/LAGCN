import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=[0.6, 0.6, 0.4, 0.4, 0.6, 0.4],
                        nargs='+',
                        help='weighted summation',
                        type=float)
    
    parser.add_argument('--slient',
                        action='store_true',
                        default=False)

    parser.add_argument('--joint',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion', default=None)
    parser.add_argument('--bone-motion', default=None)
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--prompt2', default=None)

    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        arg.alpha = [2.5, 1.5, 3.5, 1, 0.5, 1]
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        arg.alpha = [3, 2, 1, 0.5, 2, 1.5]
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        arg.alpha = [2, 2.5, 2, 0.5, 1.5, 1.5]
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(os.path.join(arg.joint_motion), 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(os.path.join(arg.bone_motion), 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(os.path.join(arg.prompt), 'rb') as r5:
        r5 = list(pickle.load(r5).items())

    with open(os.path.join(arg.prompt2), 'rb') as r6:
        r6 = list(pickle.load(r6).items())

    right_num = total_num = right_num_5 = 0

    for i in tqdm(range(len(label)), disable=arg.slient):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]
        _, r55 = r5[i]
        _, r66 = r6[i]
        r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3] + r55* arg.alpha[4] + r66* arg.alpha[5]
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
