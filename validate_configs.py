"""
모든 ablation config를 실제 학습 전에 검증.
각 config로 데이터 로딩 + 모델 빌드 + forward 1 step만 실행해서 에러를 미리 잡음.
"""
import os
import sys
import traceback
import yaml
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')


ABLATION_CONFIGS = [
    "ablation_augmentation",
    "ablation_projection",
    "ablation_batchsize",
    "ablation_temperature",
    "ablation_backbone",
    "ablation_loss",
]
BASE_CONFIG = "experiments/configs/base.yaml"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base, override):
    import copy
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def validate_one(cfg: dict) -> str | None:
    """
    단일 config 검증. 에러 없으면 None, 있으면 에러 메시지 반환.
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1) 데이터 로더
        from data.cifar10_loader import get_simclr_dataloader
        loader = get_simclr_dataloader(
            batch_size=8,           # 최소 batch로 빠르게 확인
            num_workers=0,
            augmentation=cfg.get('augmentation', 'full'),
            root=cfg.get('data_root', './data/cifar10'),
            **cfg.get('aug_kwargs', {}),
        )
        v1, v2, _ = next(iter(loader))
        v1, v2 = v1.to(device), v2.to(device)

        # 2) 모델 빌드
        from models.projection_head import build_simclr_model
        model = build_simclr_model(
            arch=cfg.get('arch', 'resnet18'),
            proj_hidden_dim=cfg.get('proj_hidden_dim', 2048),
            proj_output_dim=cfg.get('proj_output_dim', 128),
            proj_num_layers=cfg.get('proj_num_layers', 2),
            cifar_stem=cfg.get('cifar_stem', True),
        ).to(device)

        # 3) forward pass
        _, z1 = model(v1)
        _, z2 = model(v2)

        # 4) loss 계산
        from losses.nt_xent import get_loss
        loss_kwargs = dict(cfg.get('loss_kwargs', {}))
        loss_kwargs['device'] = device
        criterion = get_loss(cfg.get('loss_type', 'nt_xent'), **loss_kwargs)
        loss = criterion(z1, z2)
        loss.backward()

        return None  # 통과

    except Exception:
        return traceback.format_exc()


def main():
    base = load_yaml(BASE_CONFIG)
    results = {}   # exp_name -> 'OK' or error msg

    for group in ABLATION_CONFIGS:
        path = f"experiments/configs/{group}.yaml"
        ablations = load_yaml(path)
        if not isinstance(ablations, list):
            ablations = [ablations]

        for override in ablations:
            cfg = deep_merge(base, override)
            exp_name = cfg['exp_name']

            print(f"  checking {exp_name} ... ", end='', flush=True)
            error = validate_one(cfg)

            if error is None:
                print("OK")
                results[exp_name] = 'OK'
            else:
                print("FAIL")
                results[exp_name] = error

    # 요약 출력
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    passed = [k for k, v in results.items() if v == 'OK']
    failed = {k: v for k, v in results.items() if v != 'OK'}

    print(f"  PASSED ({len(passed)}): {', '.join(passed)}")
    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for name, err in failed.items():
            print(f"\n  [{name}]")
            # 에러의 마지막 2줄만 출력
            lines = err.strip().splitlines()
            for line in lines[-2:]:
                print(f"    {line}")
    else:
        print("\n  All configs passed! Ready to run ablations.")

    return len(failed) == 0


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
