# Project Structure

## Organization Philosophy

パイプライン指向のフラット構造．コアロジックは `src/` パッケージに責務ごとに分離し，分析・計測・ユーティリティスクリプトはルートに配置．外部モデル定義と実験出力は `.gitignore` で除外し，シェルスクリプトで実験パラメータを管理する．

## Directory Patterns

### Core Package (`src/`)

**Purpose**: PGD 攻撃パイプラインのコアロジック
**Pattern**: 1 ファイル = 1 責務（攻撃アルゴリズム，データロード，可視化，etc.）
**Example**: `pgd.py`（PGD 攻撃），`deepfool.py`（DeepFool 初期化），`pipeline.py`（オーケストレーション）

モジュール間の依存は一方向: `main -> pipeline -> {pgd, deepfool, plot_*, data_loader, model_loader} -> {dto, math_utils, logging_config}`

### Data Transfer Objects (`src/dto.py`)

**Purpose**: モジュール間のデータ受け渡し用イミュータブルオブジェクト
**Pattern**: `__slots__` 定義，型ヒント付きコンストラクタ，ビジネスロジックなし
**Example**: `ModelOps`（TF グラフ操作），`PGDBatchResult`（攻撃結果），`ExamplePanel`（描画データ）

### Analysis Scripts (root-level)

**Purpose**: 実験結果の後処理・統計分析・図表生成
**Pattern**: 独立した CLI スクリプト（`argparse` ベース），`src/` のコアロジックに依存しないか最小限の依存
**Example**: `analyze_misclassification.py`，`analyze_timing.py`，`find_common_correct_samples.py`

### Shell Scripts (root-level)

**Purpose**: 大規模バッチ実験のパラメータ管理と実行
**Pattern**: `run_{experiment_name}.sh` の命名，共通パラメータを変数で管理
**Example**: `run_all_ex100.sh`（全モデル・全初期化の組み合わせ実験）

### External Models (`model_src/`)

**Purpose**: 評価対象の事前訓練モデル（外部リポジトリからコピー）
**Note**: `.gitignore` で除外．`model.py` + チェックポイントファイルの構成

### Experiment Outputs (`outputs/`)

**Purpose**: 実験結果の保存先
**Pattern**: サブディレクトリで種類を分離: `arrays/`, `figures/`, `images/`, `metadata/`, `timing/`
**Note**: `.gitignore` で除外

### Thesis Documents (`docs/`)

**Purpose**: 卒業論文の LaTeX ソース，参考論文，進捗資料

#### LaTeX プロジェクト (`docs/卒論/`)

**Build**: `latexmk`（`latexmkrc` で設定済み，`platex` + `dvipdfmx` チェーン）
**Entry**: `main.tex` -> 各章を `\include{chapter/XX_Name}` で読み込み
**Document Class**: `qu_graduate.cls`（九州大学卒論フォーマット）

**Chapter Pattern**: `chapter/` 配下に `{番号}_{英名}.tex` で章ごとに分離
- `01_Introduction.tex` ~ `06_Conclusion.tex`: 本文
- `Appendix_*.tex`: 付録（計算コスト分析，ロス曲線の追加図）
- `07_Acknowledgement.tex`, `08_Reference.tex`: 謝辞・参考文献

**Figure Management**: 実験スクリプトの出力を `figure/` にコピーして論文に使用
- `figures_ex{N}/`, `figures_ex{N}_old/`: 実験結果のスナップショット（サンプル数 N ごとに管理）
- `outputs/`: 分析スクリプト（`analyze_misclassification.py` 等）の出力先

#### その他のドキュメント (`docs/` 直下)

- 参照論文 PDF（`参照論文_deepfool.pdf`, `参照論文_pgd.pdf`）
- 進捗報告資料（`中間発表.md`, `卒論進捗報告.md`）

## Naming Conventions

- **Directories**: ハイフン区切り（`model-src` 等），ただし既存は `model_src`, `outputs` のアンダースコア
- **Python Files**: snake_case（`plot_panel.py`, `math_utils.py`）
- **Shell Scripts**: snake_case（`run_all_ex100.sh`）
- **Classes**: PascalCase（`ModelOps`, `PGDBatchResult`, `ExamplePanel`）
- **Functions**: snake_case（`run_pgd_batch`, `build_deepfool_init`）
- **Constants**: UPPER_SNAKE_CASE（`LOGGER_NAME`, `MODEL_ORDER`）

## Import Organization

```python
# Standard library
import argparse
import os
from typing import List, Optional, Tuple

# Third-party
import numpy as np
import tensorflow as tf

# Internal (src package)
from src.dto import ModelOps, PGDBatchResult
from src.math_utils import clip_to_unit_interval, project_linf
from src.logging_config import LOGGER
```

**Rules**:
- 標準ライブラリ -> サードパーティ -> 内部モジュール の順
- `src.` プレフィックスで絶対インポート（相対インポートは不使用）
- TF 依存コードは必要に応じて遅延インポート

## Code Organization Principles

- **DTO パターン**: モジュール間のデータ受け渡しは DTO（`dto.py`）を経由し，密結合を防ぐ
- **パイプラインオーケストレーション**: `pipeline.py` が全体の流れを制御し，各モジュールは独立した機能を提供
- **TF 分離**: TensorFlow 依存コードと純粋な NumPy/Python コードを明確に分離し，テスタビリティを確保
- **明示的キャスト**: 全ての数値変換で `float()`, `int()`, `.astype()` を使用

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
