import os
import argparse

from irnlm.utils import write


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="""Split a dataset. You should give the path to the Integral Dataset.""")
    parser.add_argument("--path", type=str)
    parser.add_argument("--nsplits", type=int)

    args = parser.parse_args()
    n_splits = args.nsplits
    # Reading data
    data = open(args.path, 'r').read()
    n = len(data)
    saving_folder = os.path.dirname(args.path)
    name = os.path.basename(args.path).split('.')[0]
    print(f'The Input text contains {n} elements.')

    # Splitting data
    for index in range(n_splits):
        tmp = data[index*n//n_splits: (index+1)*n//n_splits]
        write(os.path.join(saving_folder, name+f'_split-{index+1}.txt'), tmp)