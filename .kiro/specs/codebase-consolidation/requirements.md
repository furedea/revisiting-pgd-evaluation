# Requirements Document

## Project Description (Input)
@loss_curves.pyにあるmulti-deepfool initと，@measure_timing.pyをコマンドライン引数という形でsrc以下に統合．また，find_common_correct_samples.pyやanalyze_misclassification.py, analyze_timing.pyとそれらを一括実行するrun_all_ex100.sh, run_timing_ex100.shなどもわかりやすく実行しやすくなるようにして．そしてリファクタリング．テストもしていきたいんだけど実行環境が異なるから無理そうかな

## Introduction

本仕様は，PGD 攻撃評価プロジェクトのコードベース統合・リファクタリングに関する要件を定義する．現在，Multi-DeepFool 初期化のロジックは `loss_curves.py`（レガシー単一ファイル版）に実装されており，実行時間計測は `measure_timing.py`（ルートレベルの独立スクリプト）に実装されている．これらを `src/` パッケージに CLI 引数として統合し，分析スクリプト群の使いやすさを向上させ，コード品質を改善する．

テストは全コードを対象としたいが，TF 1.x GPU 実行環境（Singularity コンテナ）が開発環境（ARM/Apple Silicon）と異なるため，TF 依存テストは環境不在時にスキップする方針とする．テスト追加は必須ではなく，ベストエフォートとする．

タイミング計測の統合にあたっては，`measure_timing.py` がアルゴリズム実行以外の余分な処理（可視化用データ保存，ログ出力等）を意図的に排除して正確な計測を行っている点に留意する．`src/` の共通コード再利用時に計測精度への悪影響を排除できない場合は，CLI 引数による「計測モード」切り替えで最低限の処理のみ実行するコードパスを設ける．

## Requirements

### Requirement 1: Multi-DeepFool 初期化の src/ パッケージ統合
**Objective:** As a 研究者, I want `loss_curves.py` にある Multi-DeepFool 初期化ロジックを `src/` パッケージに統合したい, so that 全ての初期化手法（random, deepfool, multi_deepfool, clean）を `src/main.py` の単一エントリポイントから CLI 引数で切り替えて実行できる

#### Acceptance Criteria
1. When `--init multi_deepfool` が CLI 引数として指定された場合, the PGD パイプライン shall `src/` パッケージ内の Multi-DeepFool 初期化モジュールを使用して PGD 攻撃を実行する
2. The `src/cli.py` shall `--init` 引数の選択肢に `multi_deepfool` を含む
3. When `--init multi_deepfool` が指定された場合, the パイプライン shall `loss_curves.py` と同等の `multi_deepfool_with_trace` および `run_multi_deepfool_init_pgd` の機能を提供する
4. The `src/main.py` shall 全ての初期化手法（random, deepfool, multi_deepfool, clean）を単一エントリポイントから実行できる
5. When Multi-DeepFool 初期化が統合された後, the `loss_curves.py` shall レガシーファイルとして削除される
6. The Multi-DeepFool モジュール shall `loss_curves.py` と同一の出力配列フォーマット（losses, preds, corrects の shape）を生成する

### Requirement 2: 実行時間計測の src/ パッケージ統合
**Objective:** As a 研究者, I want `measure_timing.py` の実行時間計測機能を `src/` パッケージに統合したい, so that タイミング計測を `src/` の共通モジュール（ModelOps, data_loader 等）を再利用して実行でき，コード重複を排除しつつ計測精度を維持できる

#### Acceptance Criteria
1. The 実行時間計測機能 shall `src/` パッケージ内の専用モジュールとして提供される
2. The 実行時間計測モジュール shall `src/` の既存共通モジュール（`dto.py`, `data_loader.py`, `model_loader.py`, `math_utils.py`）を再利用する
3. When `measure_timing.py` のロジックが統合された後, the ルートレベルの `measure_timing.py` shall 削除される
4. The タイミング計測 CLI shall データセット，モデル名，初期化手法，リスタート数，共通インデックスファイルを引数として受け付ける
5. The タイミング計測 shall 初期化時間，PGD 時間，合計時間をサンプルごとに JSON 形式で出力する
6. The タイミング計測 shall ウォームアップ実行を含み，安定した計測結果を提供する
7. The タイミング計測の初期化・PGD 実行コードパス shall 計測対象外の処理（可視化用データ保存，tqdm 進捗表示，詳細ログ出力等）を含まない
8. If `src/` の共通 PGD/初期化コードの再利用によって計測対象外の処理が計測区間に混入する場合, the タイミング計測モジュール shall 計測専用の軽量コードパスを独自に実装し，余分な処理を排除する
9. Where 計測専用コードパスが必要となった場合, the CLI shall `--timing` や `--mode timing` 等の引数で通常モードと計測モードを切り替えられる

### Requirement 3: 分析スクリプトの使いやすさ向上
**Objective:** As a 研究者, I want `find_common_correct_samples.py`，`analyze_misclassification.py`，`analyze_timing.py` をわかりやすく実行しやすくしたい, so that 分析ワークフロー全体を簡潔なコマンドで実行でき，パラメータの指定ミスを減らせる

#### Acceptance Criteria
1. The 分析スクリプト群 shall データセット固有のパラメータ（epsilon, alpha, model_src_dir 等）をデータセット名から自動解決する
2. The `find_common_correct_samples.py` shall `src/` パッケージの共通モジュールを再利用し，ルートレベルに維持される
3. The `analyze_misclassification.py` shall 入力ディレクトリから実験名を自動推定する
4. The `analyze_timing.py` shall 入力ディレクトリから実験名を自動推定する
5. If 必要な入力ファイルが存在しない場合, the 分析スクリプト shall 明確なエラーメッセージを表示する
6. The 分析スクリプト群 shall `--help` で使用方法と引数の説明を表示する

### Requirement 4: バッチ実行スクリプトの改善
**Objective:** As a 研究者, I want `run_all_ex100.sh` と `run_timing_ex100.sh` を統合・改善したい, so that 実験パラメータの変更が容易で，スクリプトの保守性が向上する

#### Acceptance Criteria
1. The バッチ実行スクリプト shall Multi-DeepFool 初期化の実行に `loss_curves.py` ではなく `src/main.py` を使用する
2. The バッチ実行スクリプト shall 全ての初期化手法（clean, random, deepfool, multi_deepfool）を単一エントリポイント（`src/main.py`）経由で実行する
3. The バッチ実行スクリプト shall 共通パラメータ（epsilon, alpha, 反復数等）を変数で一元管理する
4. The バッチ実行スクリプト shall 実行ログをタイムスタンプ付きで出力ディレクトリに保存する
5. The タイミング計測バッチスクリプト shall 統合後のタイミング計測モジュールを使用する

### Requirement 5: コード重複の排除とリファクタリング
**Objective:** As a 研究者, I want `loss_curves.py` と `src/` パッケージ間のコード重複を排除し，コードベース全体の品質を向上させたい, so that 保守性が高く，変更時の不整合リスクが低いコードベースになる

#### Acceptance Criteria
1. The リファクタリング後のコードベース shall `ModelOps`, `PGDBatchResult`, `ExamplePanel` 等の DTO を `src/dto.py` に一元定義する
2. The リファクタリング後のコードベース shall 数学ユーティリティ（`clip_to_unit_interval`, `project_linf`, `linf_distance`）を `src/math_utils.py` に一元定義する
3. The リファクタリング後のコードベース shall モデルロード関連ロジック（`load_model_module`, `instantiate_model`, `create_tf_session`, `restore_checkpoint`）を `src/model_loader.py` に一元定義する
4. The リファクタリング後のコードベース shall `loss_curves.py` の DTO・ユーティリティ・モデルロードの重複コピーを全て削除する
5. The リファクタリング後のコードベース shall `measure_timing.py` の DTO・ユーティリティ・モデルロードの重複コピーを全て削除する
6. The リファクタリング後のコードベース shall モジュール間の依存を一方向に保つ（`main -> pipeline -> {pgd, deepfool, multi_deepfool, plot_*, data_loader, model_loader} -> {dto, math_utils, logging_config}`）

### Requirement 6: テスト
**Objective:** As a 研究者, I want コードベース全体に対するユニットテストを追加したい, so that リファクタリング時の回帰を検出できる．ただし実行環境の制約があるため，テスト追加はベストエフォートとする

#### Acceptance Criteria
1. The テストスイート shall コードベース全体（TF 依存部分を含む）を対象とする
2. If TF が利用不可の環境の場合, the TF 依存テスト shall `pytest.mark.skipif` 等で自動的にスキップされる
3. The テストスイート shall 新規追加される Multi-DeepFool モジュールの TF 非依存部分（`compute_perturbation_to_target` 等）をテストする
4. The テストスイート shall 分析スクリプト（`analyze_misclassification.py`, `analyze_timing.py`）のファイルパース・統計計算ロジックをテストする
5. The テストスイート shall `pytest tests/` で実行できる
6. The テストスイート shall CLI 引数のパース・バリデーションロジックをテストする
7. While テスト追加がベストエフォートである間, the リファクタリング shall テストが不完全であっても進行できる

### Requirement 7: 出力互換性の維持
**Objective:** As a 研究者, I want リファクタリング後も既存の出力形式と互換性を維持したい, so that 既存の分析スクリプトや論文用図表の生成ワークフローが変更なく動作する

#### Acceptance Criteria
1. The 統合後の `src/main.py` shall `loss_curves.py` と同一のディレクトリ構造（`arrays/`, `figures/`, `metadata/`）に出力を保存する
2. The 統合後のパイプライン shall `loss_curves.py` と同一のファイル命名規則（`{dataset}_{model}_{init}_p{n}_losses.npy` 等）で出力する
3. The 統合後のパイプライン shall `analyze_misclassification.py` が期待するファイル名パターン（`*_corrects.npy`）と互換な出力を生成する
4. The 統合後のタイミング計測 shall `analyze_timing.py` が期待する JSON 形式と互換な出力を生成する
5. The 統合後のパイプライン shall メタデータファイルに Multi-DeepFool 固有のパラメータ（`df_max_iter`, `df_overshoot`）を含める
