"""
SimCLR 실험 단일 진입점.

사용법:
  # SimCLR 학습
  python run_experiment.py --mode train --config experiments/configs/base.yaml

  # Supervised baseline 학습
  python run_experiment.py --mode supervised --config experiments/configs/supervised_baseline.yaml

  # Linear probe 평가
  python run_experiment.py --mode eval --config experiments/configs/base.yaml \
      --checkpoint results/simclr_base_epoch200.pt

  # k-NN 평가
  python run_experiment.py --mode knn --config experiments/configs/base.yaml \
      --checkpoint results/simclr_base_epoch200.pt

  # t-SNE 시각화 (SimCLR + Supervised 비교)
  python run_experiment.py --mode tsne \
      --config experiments/configs/base.yaml \
      --checkpoint results/simclr_base_epoch200.pt \
      --sup_checkpoint results/supervised_resnet18_final.pt

  # Feature 분석 (collapse, separation)
  python run_experiment.py --mode analysis \
      --config experiments/configs/base.yaml \
      --checkpoint results/simclr_base_epoch200.pt

  # Ablation: 하나의 YAML에 담긴 실험 목록 일괄 실행
  python run_experiment.py --mode ablation \
      --config experiments/configs/ablation_batchsize.yaml \
      --base_config experiments/configs/base.yaml

  # 결과 요약 차트 생성
  python run_experiment.py --mode report
"""

import argparse
import copy
import os
import sys
import yaml


# ─── 헬퍼 ─────────────────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    """base config에 override의 값을 재귀적으로 덮어씀."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> dict | list:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── 각 모드 실행 함수 ─────────────────────────────────────────────────────────

def run_train(config: dict):
    sys.path.insert(0, os.path.dirname(__file__))
    from training.simclr_trainer import SimCLRTrainer
    trainer = SimCLRTrainer(config)
    trainer.train()


def run_supervised(config: dict):
    sys.path.insert(0, os.path.dirname(__file__))
    from training.supervised_trainer import SupervisedTrainer
    trainer = SupervisedTrainer(config)
    trainer.train()


def run_eval(config: dict, checkpoint: str):
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluation.linear_probe import run_linear_probe
    run_linear_probe(checkpoint, config, save_dir=config.get('save_dir', './results'))


def run_knn(config: dict, checkpoint: str):
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluation.knn_eval import run_knn_eval
    run_knn_eval(checkpoint, config, save_dir=config.get('save_dir', './results'))


def run_tsne(config: dict, checkpoint: str, sup_checkpoint: str):
    sys.path.insert(0, os.path.dirname(__file__))
    from visualization.tsne_plot import run_tsne_comparison, run_tsne_single
    if sup_checkpoint:
        sup_cfg = {'arch': config.get('arch', 'resnet18'),
                   'cifar_stem': config.get('cifar_stem', True),
                   'exp_name': 'supervised',
                   'data_root': config.get('data_root', './data/cifar10')}
        run_tsne_comparison(checkpoint, sup_checkpoint, config, sup_cfg,
                            save_dir=config.get('save_dir', './results'))
    else:
        run_tsne_single(checkpoint, config, model_type='simclr',
                        save_dir=config.get('save_dir', './results'))


def run_analysis(config: dict, checkpoint: str, model_type: str = 'simclr'):
    sys.path.insert(0, os.path.dirname(__file__))
    from visualization.feature_analysis import run_feature_analysis
    run_feature_analysis(checkpoint, config,
                         save_dir=config.get('save_dir', './results'),
                         model_type=model_type)


def _is_done(save_dir: str, exp_name: str, epochs: int) -> bool:
    """학습 완료 여부: history.json에 epochs 수만큼 loss가 기록됐는지 확인."""
    hist_path = os.path.join(save_dir, f'{exp_name}_history.json')
    if not os.path.exists(hist_path):
        return False
    import json as _json
    with open(hist_path) as f:
        hist = _json.load(f)
    return len(hist.get('train_loss', [])) >= epochs


def run_ablation(base_config_path: str, ablation_config_path: str, save_dir_override: str = None):
    """ablation YAML의 각 실험을 순차 실행. 이미 완료된 실험은 건너뜀."""
    base = load_config(base_config_path)
    ablations = load_config(ablation_config_path)

    if not isinstance(ablations, list):
        ablations = [ablations]

    for override in ablations:
        cfg = deep_merge(base, override)
        exp_name = cfg['exp_name']
        epochs   = cfg.get('epochs', 200)

        # 그룹 폴더 아래 실험별 서브폴더 생성
        # e.g. results/ablation_projection/proj_1layer/
        base_dir = save_dir_override or cfg.get('save_dir', './results')
        save_dir = os.path.join(base_dir, exp_name)
        cfg['save_dir'] = save_dir
        os.makedirs(save_dir, exist_ok=True)

        ckpt_path = os.path.join(save_dir, f'{exp_name}_best.pt')

        print(f"\n{'='*60}")
        print(f"  Ablation: {exp_name}")
        print(f"{'='*60}")

        # 이미 완료된 실험이면 학습 건너뜀
        if _is_done(save_dir, exp_name, epochs):
            print(f"  [SKIP] already done — {exp_name}")
        else:
            run_train(cfg)

        # 평가도 결과 파일 있으면 건너뜀
        probe_path = os.path.join(save_dir, f'{exp_name}_linear_probe.json')
        if os.path.exists(probe_path):
            print(f"  [SKIP] eval already done — {exp_name}")
        elif os.path.exists(ckpt_path):
            run_eval(cfg, ckpt_path)
            run_knn(cfg, ckpt_path)
            run_analysis(cfg, ckpt_path)
        else:
            print(f"  [WARN] checkpoint not found: {ckpt_path}")


def run_report(results_dir: str):
    sys.path.insert(0, os.path.dirname(__file__))
    from visualization.comparison_plots import generate_summary_report
    generate_summary_report(results_dir, results_dir)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SimCLR Experiment Runner')
    parser.add_argument('--mode', required=True,
                        choices=['train', 'supervised', 'eval', 'knn',
                                 'tsne', 'analysis', 'ablation', 'report'])
    parser.add_argument('--config',       default=None, help='실험 config YAML 경로')
    parser.add_argument('--base_config',  default='experiments/configs/base.yaml',
                        help='ablation 시 기본 config')
    parser.add_argument('--checkpoint',   default=None, help='SimCLR 체크포인트 경로')
    parser.add_argument('--sup_checkpoint', default=None, help='Supervised 모델 체크포인트 (tsne 비교용)')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--save_dir',    default=None, help='ablation 결과 저장 폴더 (base.yaml의 save_dir 덮어씀)')
    args = parser.parse_args()

    # 작업 디렉토리를 스크립트 위치로 설정
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, '.')

    if args.mode == 'report':
        run_report(args.results_dir)
        return

    if args.mode == 'ablation':
        if not args.config:
            parser.error('--config required for ablation mode')
        run_ablation(args.base_config, args.config, save_dir_override=args.save_dir)
        return

    # 나머지 모드는 config 필수
    if not args.config:
        parser.error('--config is required')

    config = load_config(args.config)
    if isinstance(config, list):
        # 목록이면 첫 번째 항목만 사용 (단일 실험)
        config = deep_merge(load_config(args.base_config), config[0])

    if args.mode == 'train':
        run_train(config)
    elif args.mode == 'supervised':
        run_supervised(config)
    elif args.mode == 'eval':
        if not args.checkpoint:
            parser.error('--checkpoint required for eval mode')
        run_eval(config, args.checkpoint)
    elif args.mode == 'knn':
        if not args.checkpoint:
            parser.error('--checkpoint required for knn mode')
        run_knn(config, args.checkpoint)
    elif args.mode == 'tsne':
        if not args.checkpoint:
            parser.error('--checkpoint required for tsne mode')
        run_tsne(config, args.checkpoint, args.sup_checkpoint)
    elif args.mode == 'analysis':
        if not args.checkpoint:
            parser.error('--checkpoint required for analysis mode')
        model_type = 'supervised' if 'supervised' in config.get('exp_name', '') else 'simclr'
        run_analysis(config, args.checkpoint, model_type=model_type)


if __name__ == '__main__':
    main()
