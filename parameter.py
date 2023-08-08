import argparse

x = '1/'


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default='0', type=str, help='Gpu id lists')
    parser.add_argument('--model_path', default='/Modeldata/fold/' + x, type=str)
    parser.add_argument('--result_path', default='/result/fold/' + x, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--model_str', default='densenet161', type=str)
    args = parser.parse_args()
    return args




