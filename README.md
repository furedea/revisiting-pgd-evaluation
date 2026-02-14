# Revisiting PGD Evaluation

誤分類速度に着目した PGD 攻撃のロバスト性評価の再検証

## Overview

PGD (Projected Gradient Descent) 攻撃に対するニューラルネットワークのロバスト性を，
複数の初期化手法（Random，DeepFool，Multi-DeepFool）を用いて評価するツールキット．

主な機能:
- PGD 攻撃の実行と損失曲線・正誤ヒートマップ・画像パネルの生成
- DeepFool / Multi-DeepFool 初期化による多様な摂動開始点の評価
- 初期化手法ごとの実行時間計測
- 誤分類率の統計分析
- モデル間比較（自然訓練，敵対的訓練など）

## Requirements

- Python 3.6.9+
- TensorFlow 1.15.5（`tf.compat.v1` Session ベース）
- NumPy, Matplotlib, tqdm, pytest

## Setup

### Docker（推奨）

```bash
# イメージのビルド
docker build --platform linux/amd64 -t pgd-test .
```

ベースイメージ `tensorflow/tensorflow:1.15.5-gpu-py3-jupyter` に
TF・NumPy・Matplotlib が同梱されており，Dockerfile では `pytest` と `tqdm` のみ追加インストールされます．

> **Note**: Apple Silicon (ARM) 上では `--platform linux/amd64` が必要です．
> TF 1.15 は AVX 命令を必要とするため，ARM ネイティブでは動作しません．

#### Docker での実験実行

```bash
# テスト実行（デフォルト CMD）
docker run --platform linux/amd64 --rm pgd-test

# PGD 攻撃の実行（CMD を上書き）
docker run --platform linux/amd64 --rm \
    -v $(pwd)/model_src:/workspace/model_src \
    -v $(pwd)/outputs:/workspace/outputs \
    pgd-test python src/main.py \
        --dataset mnist \
        --model_src_dir model_src/mnist/nat \
        --ckpt_dir model_src/mnist/nat/checkpoints \
        --out_dir outputs --exp_name ex100 \
        --epsilon 0.3 --alpha 0.01 \
        --init random --num_restarts 20 --total_iter 100

# バッチスクリプトの実行
docker run --platform linux/amd64 --rm \
    -v $(pwd)/model_src:/workspace/model_src \
    -v $(pwd)/outputs:/workspace/outputs \
    pgd-test bash run_all_ex100.sh

# GPU を使用する場合（NVIDIA Docker 必須）
docker run --platform linux/amd64 --gpus all --rm \
    -v $(pwd)/model_src:/workspace/model_src \
    -v $(pwd)/outputs:/workspace/outputs \
    pgd-test bash run_all_ex100.sh
```

`-v` でホストの `model_src/`（モデル）と `outputs/`（結果出力先）をマウントします．

### ローカル環境

Docker を使わない場合は，TF 1.15.5 をサポートする x86_64 Linux 環境で
以下をインストールしてください:

```bash
pip install tensorflow==1.15.5 tqdm pytest
```

> NumPy と Matplotlib は TF 1.15.5 の依存として自動インストールされます．

## Project Structure

```
src/                           # コアモジュール（1ファイル = 1責務）
  ├── main.py                  # メインエントリポイント
  ├── cli.py                   # CLI 引数パース
  ├── pipeline.py              # パイプライン制御
  ├── pgd.py                   # PGD 攻撃アルゴリズム
  ├── deepfool.py              # DeepFool 初期化
  ├── multi_deepfool.py        # Multi-DeepFool 初期化
  ├── timing.py                # タイミング計測用軽量コードパス
  ├── timing_cli.py            # タイミング計測 CLI
  ├── dataset_config.py        # データセット固有パラメータ定義
  ├── data_loader.py           # データ読み込み
  ├── model_loader.py          # モデル読み込み
  ├── dto.py                   # データ転送オブジェクト
  ├── math_utils.py            # 数学ユーティリティ
  ├── plot_setup.py            # 可視化設定
  ├── plot_save.py             # 可視化保存
  └── logging_config.py        # ロギング設定

find_common_correct_samples.py # 共通正解サンプルのインデックス生成
analyze_misclassification.py   # 誤分類率の統計分析
analyze_timing.py              # タイミング結果の統計分析
run_all_ex100.sh               # バッチ実行（全モデル × 全初期化手法）
run_timing_ex100.sh            # タイミング計測バッチ実行

tests/                         # テストスイート
outputs/                       # 実験結果出力（.gitignore）
model_src/                     # 事前訓練モデル（.gitignore）
docs/卒論/                     # 論文 LaTeX ソース
```

## Usage

### ワークフロー概要

```
1. 共通正解サンプル特定  →  2. PGD 攻撃実行  →  3. タイミング計測  →  4. 分析
   find_common_correct_       src/main.py          src.timing_cli        analyze_*.py
   samples.py
```

### 1. 共通正解サンプルの特定

全モデルが正しく分類するサンプルのインデックスを事前に計算します．
この結果は PGD 攻撃とタイミング計測の両方で `--common_indices_file` として使用します:

```bash
python find_common_correct_samples.py \
    --dataset mnist \
    --models nat adv nat_and_adv weak_adv \
    --n_samples 100
```

出力: `docs/common_correct_indices_mnist_n100.json`

### 2. PGD 攻撃の実行

`src/main.py` が全初期化手法を統一的に扱うメインエントリポイントです:

```bash
python src/main.py \
    --dataset mnist \
    --model_src_dir model_src/mnist/nat \
    --ckpt_dir model_src/mnist/nat/checkpoints \
    --out_dir outputs \
    --exp_name ex100 \
    --epsilon 0.3 \
    --alpha 0.01 \
    --init random \
    --num_restarts 20 \
    --total_iter 100 \
    --n_examples 100 \
    --common_indices_file docs/common_correct_indices_mnist_n100.json
```

#### 初期化手法の選択

| `--init` | 説明 |
|---|---|
| `random` | L-inf ボール内のランダム初期化（デフォルト） |
| `deepfool` | DeepFool による最近接決定境界方向への初期化 |
| `multi_deepfool` | 全ターゲットクラスへの DeepFool で多様な初期点を生成 |
| `clean` | 元画像から開始（ベースライン比較用） |

#### DeepFool / Multi-DeepFool 固有オプション

```bash
--df_max_iter 50       # DeepFool 反復回数（デフォルト: 50）
--df_overshoot 0.02    # オーバーシュート係数（デフォルト: 0.02）
--df_jitter 0.0        # 初期点へのジッタ（デフォルト: 0.0）
```

### 3. タイミング計測

初期化 + PGD 実行時間をサンプルごとに計測し JSON で出力します．
`src/timing_cli.py` をモジュールとして実行します
（`-m` フラグにより `src` パッケージ内のインポートが正しく解決されます）:

```bash
python -m src.timing_cli \
    --dataset mnist \
    --model nat \
    --init random \
    --num_restarts 1 \
    --common_indices_file docs/common_correct_indices_mnist_n100.json \
    --out_dir outputs \
    --exp_name timing_ex100
```

> `--epsilon` と `--alpha` は `--dataset` から自動解決されるため省略可能です．

出力例（JSON）:

```json
{
  "dataset": "mnist",
  "model": "nat",
  "init": "random",
  "num_restarts": 1,
  "indices": [3, 17, 42],
  "results": [
    {"init": 0.001, "pgd": 0.523, "total": 0.524},
    {"init": 0.001, "pgd": 0.518, "total": 0.519}
  ]
}
```

### 4. バッチ実行

全モデル・全初期化手法をまとめて実行:

```bash
# PGD 攻撃（全組み合わせ）
bash run_all_ex100.sh

# タイミング計測（全組み合わせ）
bash run_timing_ex100.sh
```

バッチスクリプトは共通正解インデックスが未生成の場合，自動で生成します．

### 5. 分析

```bash
# 誤分類率の分析
python analyze_misclassification.py \
    --input_dir outputs/ex100 \
    --out_dir outputs/analysis

# タイミング結果の分析
python analyze_timing.py \
    --input_dir outputs/timing_ex100 \
    --out_dir outputs/timing_analysis
```

## Testing

### テストの実行

```bash
# Docker 上で全テスト実行（推奨）
docker build --platform linux/amd64 -t pgd-test .
docker run --platform linux/amd64 --rm pgd-test

# 特定のテストファイルのみ実行
docker run --platform linux/amd64 --rm pgd-test pytest tests/test_dto.py -v

# ローカル環境で実行（TF 依存テストは自動スキップ）
pytest tests/ -v
```

### TF 依存テストの自動スキップ

TF 1.15 は AVX 命令を必要とし，ARM エミュレーション環境では segfault します．
`tests/conftest.py` が `/proc/cpuinfo` の AVX フラグを検出し，
TF を安全にインポートできない環境では `@pytest.mark.requires_tf` 付きテストを自動スキップします．

環境変数 `SKIP_TF_TESTS=1` で強制スキップも可能です:

```bash
SKIP_TF_TESTS=1 pytest tests/ -v
```

### テストファイル一覧

| テストファイル | 対象モジュール | TF 依存 |
|---|---|---|
| `test_dto.py` | `src/dto.py` — データ転送オブジェクト | No |
| `test_math_utils.py` | `src/math_utils.py` — 数学ユーティリティ | No |
| `test_cli.py` | `src/cli.py` — CLI 引数パース | No |
| `test_dataset_config.py` | `src/dataset_config.py` — データセット設定 | No |
| `test_logging_config.py` | `src/logging_config.py` — ロギング設定 | No |
| `test_pgd.py` | `src/pgd.py` — PGD 攻撃（NumPy 部分） | No |
| `test_multi_deepfool.py` | `src/multi_deepfool.py` — 摂動計算（NumPy）+ 統合テスト | 一部 |
| `test_pipeline.py` | `src/pipeline.py` — パイプライン制御 | No (mock) |
| `test_plot_setup.py` | `src/plot_setup.py` — 可視化設定 | No |
| `test_plot_save.py` | `src/plot_save.py` — 可視化保存 | No (mock) |
| `test_timing.py` | `src/timing.py` — タイミング計測コードパス | No (mock) |
| `test_timing_cli.py` | `src/timing_cli.py` — タイミング計測 CLI | No (mock) |
| `test_analyze_misclassification.py` | `analyze_misclassification.py` — 誤分類分析 | No |
| `test_analyze_timing.py` | `analyze_timing.py` — タイミング分析 | No |
| `test_find_common_correct_samples.py` | `find_common_correct_samples.py` | No |
| `test_run_all_script.py` | `run_all_ex100.sh` — バッチスクリプト構文検証 | No |
| `test_batch_scripts.py` | `run_timing_ex100.sh` — バッチスクリプト構文検証 | No |
| `test_legacy_removal.py` | レガシーファイル削除の検証 | No |
| `test_conftest.py` | `conftest.py` — TF スキップ機構の検証 | 一部 |

## Output Format

実験結果は以下のディレクトリ構造で出力されます:

```
outputs/{exp_name}/{dataset}/{model}/{init}/
  ├── arrays/          # NumPy 配列（losses, preds, corrects）
  ├── figures/         # 損失曲線・ヒートマップ画像
  └── metadata/        # 実験パラメータの JSON
```

## Supported Datasets

| Dataset | Epsilon | Alpha | Classes |
|---|---|---|---|
| MNIST | 0.3 | 0.01 | 10 |
| CIFAR-10 | 8/255 | 2/255 | 10 |

## License

Kyushu University - Thesis Research Project
