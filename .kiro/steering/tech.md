# Technology Stack

## Architecture

CLI ベースのパイプラインアーキテクチャ．メインエントリポイントから CLI 引数解析 -> データロード -> モデルロード -> PGD 攻撃実行 -> 可視化・保存の一方向パイプラインで処理する．TensorFlow 1.x の Session ベース API を使用し，外部モデル定義（`model_src/`）を動的インポートする設計．

## Core Technologies

- **Language**: Python 3.6.9+ (TF 1.15.5 互換)
- **Framework**: TensorFlow 1.x (`tf.compat.v1` API)
- **Visualization**: Matplotlib (Agg バックエンド)
- **Compute**: NumPy (数値計算・配列操作)

## Key Libraries

- **TensorFlow 1.15.5**: モデルロード・推論・勾配計算（Session ベース API）
- **NumPy**: 攻撃アルゴリズム・L-inf 射影・配列操作
- **Matplotlib**: ロス曲線・正誤ヒートマップ・画像パネルの描画
- **tqdm**: PGD イテレーションの進捗表示
- **pytest**: テストフレームワーク

## Development Standards

### Type Safety

- 型ヒント（`typing` モジュール）を全関数シグネチャに付与
- `Any` は TensorFlow オブジェクト（Session, Tensor 等）のみ許容
- 数値型は明示的キャスト（`np.float32`, `np.int64`）で一貫性を保つ

### Code Quality

- docstring は全モジュール・全関数に英語で記述
- `__slots__` を DTO クラスに使用してメモリ効率と属性制限を確保
- ロガーはモジュールレベルの `LOGGER` インスタンスで統一

### Testing

- pytest を使用
- TensorFlow 非依存部分を優先的にテスト（ARM 環境で TF 1.x が動作しないため）
- DTO の `from_model` 等 TF 依存部分は遅延インポートで分離

## Development Environment

### Required Tools

- Python 3.6.9+
- TensorFlow 1.15.5（GPU 環境推奨）
- Singularity コンテナ（大規模実験用）
- LaTeX（platex + dvipdfmx）: 卒業論文のビルド（`latexmk` で自動化）

### Common Commands

```bash
# Main pipeline execution
python -m src.main --dataset mnist --model_src_dir model_src/mnist --ckpt_dir model_src/mnist/models/nat --out_dir outputs --exp_name test --epsilon 0.3 --alpha 0.01

# Analysis scripts (root-level)
python analyze_misclassification.py --input_dir outputs/arrays/run_all_ex100/ --out_dir outputs
python analyze_timing.py --input_dir outputs/timing/timing_ex100 --out_dir outputs

# Tests
pytest tests/

# Batch experiments (shell scripts)
bash run_all_ex100.sh

# Thesis build (docs/卒論/)
cd docs/卒論 && latexmk main.tex
```

## Key Technical Decisions

- **TF 1.x Session API**: 評価対象モデル（Madry et al. のチャレンジモデル）が TF 1.x で定義されているため，`tf.compat.v1` を使用
- **動的モデルインポート**: `model_src/` 配下の外部 `model.py` を `importlib` で動的にロードし，異なるモデルアーキテクチャに対応
- **遅延 TF インポート**: `dto.py` の `ModelOps.from_model` 等で TF を遅延インポートし，ARM (Apple Silicon) 環境でもテスト可能に
- **明示的な数値型キャスト**: `float()`, `int()`, `.astype()` を徹底し，TF/NumPy 間の型不整合を防止
- **Agg バックエンド**: ヘッドレスサーバー（Singularity コンテナ）での実行に対応

---
_Document standards and patterns, not every dependency_
