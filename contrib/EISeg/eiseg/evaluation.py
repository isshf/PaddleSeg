import sys
import pickle
import argparse
from pathlib import Path

import cv2
import paddle
import numpy as np

sys.path.insert(0, '.')
from util import evaluate_util
from util.exp import load_config_file
from util.vis import draw_probmap, draw_with_blend_and_clicks
from inference.predictor import get_predictor
from inference.evaluation import evaluate_dataset
from models import HRNet18_OCR64, HRNet18s_OCR48
from model.is_hrnet_model import HRNetModel
from util import MODELS

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', choices=['HRNet18_OCR64', 'HRNet18s_OCR48'],
                        help='Select model to evaluate. Possible choices: HRNet18_OCR64, HRNet18s_OCR48', default='HRNet18s_OCR48')

    parser.add_argument('--checkpoint', type=str, default='',
                                   help='The checkpoints for selected model.')

    parser.add_argument('--datasets', type=str, default='GrabCut',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC,Human, GrabCut,Berkeley,DAVIS')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target_iou', type=float, default=0.90,
                                  help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument('--iou_analysis', action='store_true', default=False,
                                  help='Plot mIoU(number of clicks) with target_iou=1.0.')

    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--min-n-clicks', type=int, default=1,
                        help='Minimum number of clicks for the evaluation.')
    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--clicks_limit', type=int, default=None)
    parser.add_argument('--eval_mode', type=str, default='cvpr',
                        help='Possible choices: cvpr, fixed<number> (e.g. fixed400, fixed600).')

    parser.add_argument('--save_ious', action='store_true', default=False)
    parser.add_argument('--print_ious', action='store_true', default=False)
    parser.add_argument('--vis_preds', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default=None,
                        help='The model name that is used for making plots.')
    parser.add_argument('--config_path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--logs_path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')

    args = parser.parse_args()

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)

    return args, cfg


def main():
    args, cfg = parse_args()

    logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = True
    print_header = single_model_eval
    for dataset_name in args.datasets.split(','):
        dataset = evaluate_util.get_dataset(dataset_name, cfg)

        if args.model_name in ["HRNet18s_OCR48", "HRNet18_OCR64"]:
            eiseg = MODELS[args.model_name]()
        else:
            raise Exception("only support HRNet18s_OCR48 or HRNet18_OCR64")
            
        eiseg.load_param(args.checkpoint)
        model = eiseg.model
        predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name)
        predictor = get_predictor(model, 'NoBRS',
                                  predictor_params=predictor_params,
                                  zoom_in_params=zoomin_params)

        vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh) if args.vis_preds else None
        dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                           max_iou_thr=args.target_iou,
                                           min_clicks=args.min_n_clicks,
                                           max_clicks=args.n_clicks,
                                           callback=vis_callback)
        row_name = 'NoBRS'
        if args.iou_analysis:
            save_iou_analysis_data(args, dataset_name, logs_path,
                                   logs_prefix, dataset_results,
                                   model_name=args.model_name)

        save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                     save_ious=single_model_eval and args.save_ious,
                     single_model_eval=single_model_eval,
                     print_header=print_header)
        print_header = False


def get_predictor_and_zoomin_params(args, dataset_name):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    if args.eval_mode == 'cvpr':
        zoom_in_params = {
            'target_size': 600 if dataset_name == 'DAVIS' else 400
        }
    elif args.eval_mode.startswith('fixed'):
        crop_size = int(args.eval_mode[5:])
        zoom_in_params = {
            'skip_clicks': -1,
            'target_size': (crop_size, crop_size)
        }
    else:
        raise NotImplementedError

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    logs_path = args.logs_path / 'others'/ args.model_name

    return logs_path, logs_prefix


def save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                 save_ious=False, print_header=True, single_model_eval=False):
    all_ious, elapsed_time = dataset_results
    mean_spc, mean_spi = evaluate_util.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, over_max_list = evaluate_util.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)

    row_name = 'last' if row_name == 'last_checkpoint' else row_name
    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = evaluate_util.get_results_table(noc_list, over_max_list, row_name, dataset_name,
                                                mean_spc, elapsed_time, args.n_clicks,
                                                model_name=model_name)

    if args.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)
        miou_str = ' '.join([f'mIoU@{click_id}={mean_ious[click_id - 1]:.2%};'
                             for click_id in [1, 2, 3, 5, 10, 20] if click_id <= min_num_clicks])
        table_row += '; ' + miou_str
    else:
        target_iou_int = int(args.target_iou * 100)
        if target_iou_int not in [80, 85, 90]:
            noc_list, over_max_list = evaluate_util.compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                               max_clicks=args.n_clicks)
            table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
            table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)

    if save_ious:
        ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.eval_mode}_NoBRS_{args.n_clicks}.pkl', 'wb') as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
        if not single_model_eval:
            name_prefix += f'{dataset_name}_'

    log_path = logs_path / f'{name_prefix}{args.eval_mode}_NoBRS_{args.n_clicks}.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')


def save_iou_analysis_data(args, dataset_name, logs_path, logs_prefix, dataset_results, model_name=None):
    all_ious, _ = dataset_results

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
    name_prefix += dataset_name + '_'
    if model_name is None:
        model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    pkl_path = logs_path / f'plots/{name_prefix}{args.eval_mode}_NoBRS_{args.n_clicks}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump({
            'dataset_name': dataset_name,
            'model_name': f'{model_name}_NoBRS',
            'all_ious': all_ious
        }, f)


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list):
        sample_path = save_path / f'{sample_id}_{click_indx}.jpg'
        prob_map = draw_probmap(pred_probs)
        image_with_mask = draw_with_blend_and_clicks(image, pred_probs > prob_thresh, clicks_list=clicks_list)
        cv2.imwrite(str(sample_path), np.concatenate((image_with_mask, prob_map), axis=1)[:, :, ::-1])

    return callback


if __name__ == '__main__':
    main()

