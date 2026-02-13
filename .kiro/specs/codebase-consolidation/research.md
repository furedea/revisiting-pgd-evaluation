# Research & Design Decisions

---
**Purpose**: コードベース統合リファクタリングに伴うディスカバリ調査の記録．
**Usage**: design.md の根拠となる調査ログ．
---

## Summary
- **Feature**: `codebase-consolidation`
- **Discovery Scope**: Extension（既存 `src/` パッケージへの統合）
- **Key Findings**:
  1. `loss_curves.py` と `src/` の間で DTO・ユーティリティ・モデルロード・PGD 実行が完全に重複しており，Multi-DeepFool 固有ロジック（`compute_perturbation_to_target`, `deepfool_single_target`, `multi_deepfool_with_trace`, `run_multi_deepfool_init_pgd`）のみが `loss_curves.py` に固有
  2. `measure_timing.py` は計測精度のために tqdm・ログ・可視化データ保存を排除した軽量 PGD コードパスを独自実装しており，`src/pgd.py` の `run_pgd_batch` をそのまま再利用すると計測精度が劣化する
  3. 分析スクリプト（`analyze_misclassification.py`, `analyze_timing.py`）は純粋な NumPy/Python ロジックで TF 非依存であり，テスタビリティが高い

## Research Log

### loss_curves.py と src/ の重複分析
- **Context**: Requirement 1, 5 に関連．`loss_curves.py`（約 1549 行）と `src/`（15 ファイル）の重複範囲を特定
- **Findings**:
  - DTO（`ModelOps`, `InitSanityMetrics`, `PGDBatchResult`, `ExamplePanel`）: 完全重複，ただし `loss_curves.py` 版は `x_df_endpoints`, `x_init_rank`, `test_idx` フィールドが追加
  - ユーティリティ（`linf_distance`, `project_linf`, `clip_to_unit_interval`）: 完全重複
  - モデルロード（`load_model_module`, `instantiate_model`, `create_tf_session`, `restore_checkpoint`）: 完全重複
  - PGD 実行: `loss_curves.py` 版は `run_pgd_batch` のインターフェースが異なる（`num_iter` vs `total_iter`, `init` 引数なし）
  - Multi-DeepFool 固有: `compute_perturbation_to_target`, `deepfool_single_target`, `multi_deepfool_with_trace`, `run_multi_deepfool_init_pgd` は `loss_curves.py` のみに存在
- **Implications**: Multi-DeepFool 固有関数を新規モジュール `src/multi_deepfool.py` に移行し，DTO に不足フィールドを追加する

### measure_timing.py の計測精度要件
- **Context**: Requirement 2.7, 2.8, 2.9 に関連．計測精度を維持しつつ `src/` の共通コードを再利用できるか調査
- **Findings**:
  - `measure_timing.py` は `run_pgd` 関数を独自実装（tqdm なし，ログなし，loss/pred 記録なし）
  - `src/pgd.py` の `run_pgd_batch` は各イテレーションで `losses`, `preds` 配列への書き込み + tqdm 表示を含む
  - `src/pgd.py` の初期化パス `build_initial_points` は `measure_timing.py` の `random_init`, `deepfool_init`, `multi_deepfool_init` と類似するが，`deepfool`/`multi_deepfool` のタイミング用軽量版は存在しない
  - `time.perf_counter()` による計測で，余分な処理は計測値にノイズを加える
- **Implications**: `src/timing.py` に計測専用の軽量 PGD 関数を配置し，`src/` の共通モジュール（DTO, math_utils, model_loader, data_loader）は再利用しつつ，PGD ループと初期化関数は計測用に最適化した版を持つ

### 分析スクリプトのパラメータ自動解決
- **Context**: Requirement 3.1 に関連．データセット固有パラメータの自動解決方法を調査
- **Findings**:
  - `measure_timing.py` は既にデータセット名からパラメータを解決: `mnist -> eps=0.3, alpha=0.01, model_src_dir=model_src/mnist_challenge`
  - `run_all_ex100.sh` は変数でパラメータ管理
  - `analyze_misclassification.py` と `analyze_timing.py` はパラメータ自動解決なし（入力ディレクトリから実験名を推定するのみ）
- **Implications**: `src/dataset_config.py` にデータセット固有パラメータの解決ロジックを集約し，全スクリプトから参照する

### 既存テストの状態
- **Context**: Requirement 6 に関連．既存テストの状態と拡張方針を調査
- **Findings**:
  - `tests/` に 6 テストファイル（`test_cli.py`, `test_dto.py`, `test_math_utils.py`, `test_logging_config.py`, `test_pgd.py`, `test_plot_setup.py`）
  - `test_cli.py` は旧インターフェース（`format_indices_part`, `steps` 引数等）を参照しており，現行 `src/cli.py` と不整合
  - `conftest.py` は `PROJECT_ROOT` を `sys.path` に追加するのみ
  - TF 依存テストのスキップ機構は未実装
- **Implications**: 既存テストの修正 + 新規テスト追加が必要．`conftest.py` に `pytest.mark.skipif` 用のフィクスチャを定義する

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| モジュール追加 | `src/` に `multi_deepfool.py`, `timing.py`, `dataset_config.py` を追加 | 既存パターンとの一貫性，最小限の変更 | 計測用軽量コードパスで一部重複が残る | 採用 |
| パイプライン抽象化 | PGD 実行を Strategy パターンで抽象化し，計測モードをフラグで切り替え | 重複ゼロ | 過度な抽象化，TF 1.x Session API との相性悪い | 不採用 |

## Design Decisions

### Decision: Multi-DeepFool モジュールの配置
- **Context**: `loss_curves.py` の Multi-DeepFool 固有ロジックを `src/` に移行する方法
- **Alternatives Considered**:
  1. `src/deepfool.py` に追加 -- 1ファイルが肥大化
  2. `src/multi_deepfool.py` として独立 -- 責務が明確
- **Selected Approach**: `src/multi_deepfool.py` として独立モジュールを作成
- **Rationale**: 単一責務原則に従い，DeepFool（単一境界）と Multi-DeepFool（複数ターゲット）は異なる概念
- **Trade-offs**: モジュール数が増加するが，`compute_perturbation_to_target` は `src/deepfool.py` にも移動可能な共通関数

### Decision: タイミング計測の計測専用コードパス
- **Context**: `src/pgd.py` の `run_pgd_batch` は計測対象外の処理（tqdm, 配列記録）を含む
- **Alternatives Considered**:
  1. `run_pgd_batch` にフラグ引数を追加して計測モードを切り替え
  2. 計測専用の軽量関数を `src/timing.py` に配置
- **Selected Approach**: `src/timing.py` に計測専用コードパスを配置
- **Rationale**: 計測精度への悪影響を完全に排除するため，分離した方が安全．`run_pgd_batch` のインターフェース変更は既存の動作に影響するリスクがある
- **Trade-offs**: PGD ループの軽微な重複が生じるが，計測精度の信頼性を優先

### Decision: DTO フィールドの拡張
- **Context**: `loss_curves.py` の `PGDBatchResult` には `x_df_endpoints`, `x_init_rank` フィールドがあるが `src/dto.py` にはない
- **Selected Approach**: `src/dto.py` の `PGDBatchResult` に `x_df_endpoints` と `x_init_rank` を Optional フィールドとして追加
- **Rationale**: 既存の random/deepfool init では None のまま，multi_deepfool init でのみ値が設定される
- **Trade-offs**: DTO のフィールド数が増加するが，後方互換性を維持

### Decision: dataset_config モジュール
- **Context**: データセット固有のパラメータ（epsilon, alpha, model_src_dir）が複数ファイルに散在
- **Selected Approach**: `src/dataset_config.py` に `DatasetConfig` DTO と `resolve_dataset_config` 関数を配置
- **Rationale**: DRY 原則に従い，パラメータ定義を一元化

## Risks & Mitigations
- **出力互換性の破壊**: Multi-DeepFool 統合後の出力配列 shape が `loss_curves.py` と異なる可能性 -- テストで shape を検証
- **計測精度の劣化**: `src/` の共通コード再利用で計測にノイズが混入 -- 計測専用コードパスで回避
- **テスト不整合**: 既存テスト(`test_cli.py`) が旧インターフェースを参照 -- テスト修正を先行
- **TF 依存テストの環境差異**: ARM 環境で TF 1.x が動作しない -- `pytest.mark.skipif` で自動スキップ

## References
- 既存コードベース: `src/`, `loss_curves.py`, `measure_timing.py`
- Steering documents: `.kiro/steering/product.md`, `tech.md`, `structure.md`
