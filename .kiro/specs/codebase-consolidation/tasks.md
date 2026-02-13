# Implementation Plan

- [ ] 1. 共通基盤の拡張（DTO・データセット設定・CLI）
- [x] 1.1 DTO の Multi-DeepFool 対応フィールド追加
  - `PGDBatchResult` に DeepFool エンドポイント配列と最大ロス初期点インデックスを Optional フィールドとして追加する
  - `ExamplePanel` に最大ロスインデックスとテストサンプルインデックスを Optional フィールドとして追加する
  - 既存の random/deepfool init では None のまま動作することを確認する
  - _Requirements: 1.6, 5.1_

- [x] 1.2 (P) データセット固有パラメータの一元管理モジュールを作成
  - MNIST と CIFAR-10 のデータセット固有パラメータ（epsilon, alpha, model_src_dir）を定義する
  - データセット名からパラメータを解決する関数を提供する
  - 不明なデータセット名に対して ValueError を送出する
  - _Requirements: 3.1, 5.1_

- [x] 1.3 CLI に Multi-DeepFool 初期化の選択肢を追加
  - `--init` 引数の選択肢に `multi_deepfool` を追加する
  - Multi-DeepFool 固有パラメータ（`df_max_iter`, `df_overshoot`）のバリデーションを追加する
  - ファイル命名規則とタイトル生成に Multi-DeepFool 対応を追加する
  - 全初期化手法（random, deepfool, multi_deepfool, clean）が単一エントリポイントから選択可能であることを確認する
  - _Requirements: 1.2, 1.4_

- [ ] 2. Multi-DeepFool 初期化モジュールの実装
- [x] 2.1 Multi-DeepFool 固有ロジックの移行
  - ターゲットクラスへの摂動計算（勾配差分とロジット差分から最小ノルム摂動を求める）を実装する
  - 単一ターゲットに対する DeepFool 反復を実装する
  - 正解ラベル以外の全ターゲットラベルに対して DeepFool を実行し，(K-1) 個の多様な初期点を生成するトレース付き関数を実装する
  - 共通モジュール（DTO, 数学ユーティリティ）を再利用し，重複コードを排除する
  - _Requirements: 1.1, 1.3, 5.4_

- [x] 2.2 Multi-DeepFool 初期化付き PGD 統合実行関数の実装
  - Multi-DeepFool で生成した初期点から PGD 攻撃を実行し，全リスタートの損失・予測・正誤を記録する関数を実装する
  - `num_restarts` がクラス数 - 1 を超える場合の `ValueError` バリデーションを実装する
  - 出力配列のフォーマット（losses, preds, corrects の shape）がレガシー版と同一であることを保証する
  - _Requirements: 1.1, 1.3, 1.6, 5.4_

- [ ] 3. パイプラインへの Multi-DeepFool 統合と出力互換性の確保
- [x] 3.1 パイプラインに Multi-DeepFool init ブランチを追加
  - パイプラインのサンプル処理関数に `multi_deepfool` 初期化ブランチを追加し，Multi-DeepFool モジュールの統合実行関数を呼び出す
  - Multi-DeepFool 固有のパラメータ（`df_max_iter`, `df_overshoot`）をメタデータに含める
  - ExamplePanel に Multi-DeepFool 固有フィールド（最大ロスインデックス，テストサンプルインデックス）を設定する
  - _Requirements: 1.1, 7.1, 7.5_

- [x] 3.2 メタデータ・保存処理の Multi-DeepFool 対応
  - メタデータフォーマット関数の条件を Multi-DeepFool に拡張する
  - 出力ディレクトリ構造（arrays/, figures/, metadata/）がレガシー版と同一であることを確認する
  - ファイル命名規則がレガシー版と互換であることを確認する
  - `*_corrects.npy` が分析スクリプトの期待するパターンと互換であることを確認する
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [x] 3.3 レガシーファイル `loss_curves.py` の削除
  - 統合完了後に `loss_curves.py` を削除する
  - 削除後もパイプライン全体が正常に動作することを確認する
  - _Requirements: 1.5, 5.4_

- [ ] 4. タイミング計測モジュールの実装
- [x] 4.1 計測専用の軽量初期化・PGD コードパスの実装
  - ランダム初期化の計測専用版（ログ・進捗表示なし）を実装する
  - DeepFool 初期化の計測専用版（ログ・配列記録なし）を実装する
  - Multi-DeepFool 初期化の計測専用版（ログ・配列記録なし）を実装する
  - PGD ループの計測専用版（tqdm・ログ・損失/予測記録なし）を実装する
  - 計測区間に計測対象外の処理が混入しないことを保証する
  - 共通モジュール（DTO, 数学ユーティリティ, モデルローダー, データローダー）を再利用する
  - _Requirements: 2.1, 2.2, 2.7, 2.8, 5.5_

- [x] 4.2 サンプル単位の計測関数と実験全体の実行関数の実装
  - 単一サンプルに対する初期化時間・PGD 時間・合計時間を `perf_counter` で計測する関数を実装する
  - ウォームアップ実行を含む実験全体の実行関数を実装する
  - 計測結果をサンプルごとの辞書リストとして返却する
  - _Requirements: 2.1, 2.5, 2.6_

- [x] 4.3 タイミング計測の CLI エントリポイントの実装
  - データセット，モデル名，初期化手法，リスタート数，共通インデックスファイルを引数として受け付ける CLI を実装する
  - データセット固有パラメータの自動解決を統合する
  - 計測結果を JSON 形式で出力する（レガシー版と互換のフォーマット）
  - `python -m src.timing_cli` で実行可能にする
  - `--help` で使用方法と引数の説明を表示する
  - _Requirements: 2.4, 2.5, 2.9, 7.4_

- [x] 4.4 レガシーファイル `measure_timing.py` の削除
  - 統合完了後に `measure_timing.py` を削除する
  - 削除後もタイミング計測が正常に動作することを確認する
  - _Requirements: 2.3, 5.5_

- [ ] 5. 分析スクリプトの使いやすさ向上
- [x] 5.1 (P) `find_common_correct_samples.py` の共通モジュール再利用
  - `src/` パッケージの共通モジュール（データローダー, モデルローダー, DTO）を再利用するようリファクタリングする
  - 入力ファイル不在時に明確なエラーメッセージを表示する
  - `--help` で使用方法と引数の説明を表示する
  - _Requirements: 3.2, 3.5, 3.6_

- [x] 5.2 (P) `analyze_misclassification.py` の改善
  - 入力ディレクトリから実験名を自動推定する機能を追加する
  - データセット固有パラメータの自動解決を統合する
  - 入力ファイル不在時に明確なエラーメッセージを表示する
  - `--help` で使用方法と引数の説明を表示する
  - _Requirements: 3.1, 3.3, 3.5, 3.6_

- [x] 5.3 (P) `analyze_timing.py` の改善
  - 入力ディレクトリから実験名を自動推定する機能を追加する
  - データセット固有パラメータの自動解決を統合する
  - 入力ファイル不在時に明確なエラーメッセージを表示する
  - `--help` で使用方法と引数の説明を表示する
  - _Requirements: 3.1, 3.4, 3.5, 3.6_

- [ ] 6. バッチ実行スクリプトの改善
- [x] 6.1 PGD 実行バッチスクリプトの統合
  - Multi-DeepFool の実行を `loss_curves.py` から `src/main.py --init multi_deepfool` に切り替える
  - 全初期化手法を単一エントリポイント経由で実行するように統一する
  - 共通パラメータ（epsilon, alpha, 反復数等）を変数で一元管理する
  - 実行ログをタイムスタンプ付きで出力ディレクトリに保存する
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 6.2 タイミング計測バッチスクリプトの統合
  - `measure_timing.py` の呼び出しを `python -m src.timing_cli` に切り替える
  - CLI 引数の互換性を維持する
  - _Requirements: 4.5_

- [ ] 7. コード重複の排除と依存方向の検証
- [x] 7.1 重複コードの最終クリーンアップ
  - DTO・ユーティリティ・モデルロードが `src/` の各モジュールに一元定義されていることを検証する
  - `loss_curves.py` と `measure_timing.py` の重複コピーが全て削除されていることを確認する
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7.2 モジュール間の依存方向の検証
  - モジュール間の依存が一方向（main -> pipeline -> {pgd, deepfool, multi_deepfool, plot_*, data_loader, model_loader} -> {dto, math_utils, logging_config}）であることを検証する
  - 循環依存が存在しないことを確認する
  - _Requirements: 5.6_

- [ ] 8. テストスイートの拡充
- [x] 8.1 テスト基盤の整備
  - TF 不可環境での自動スキップ機構（`pytest.mark.skipif`）を conftest.py に追加する
  - `pytest tests/` で全テストが実行可能であることを確認する
  - _Requirements: 6.2, 6.5_

- [x] 8.2 (P) Multi-DeepFool モジュールの TF 非依存テスト
  - 摂動計算関数の数値計算を NumPy のみでテストする
  - 入力・出力の shape と値の妥当性を検証する
  - _Requirements: 6.3_

- [x] 8.3 (P) CLI 引数パース・バリデーションのテスト
  - `--init multi_deepfool` を含む引数パースの正常系・異常系をテストする
  - Multi-DeepFool 固有パラメータのバリデーションをテストする
  - _Requirements: 6.6_

- [x] 8.4 (P) データセット設定モジュールのテスト
  - 全データセットのパラメータ解決が正しい値を返すことをテストする
  - 不明なデータセット名に対する ValueError をテストする
  - _Requirements: 6.1_

- [x] 8.5 (P) 分析スクリプトのテスト
  - `analyze_misclassification.py` のファイル名パース・統計計算ロジックをテストする
  - `analyze_timing.py` のタイミング結果ロード・統計計算ロジックをテストする
  - _Requirements: 6.4_

- [x]*8.6 (P) TF 依存の統合テスト（ベストエフォート）
  - Multi-DeepFool パイプラインの出力 shape 検証テストを追加する（`@requires_tf`）
  - タイミング計測の戻り値が正の辞書であることを検証するテストを追加する（`@requires_tf`）
  - TF 不可環境では自動スキップされることを確認する
  - テスト追加はベストエフォートとし，不完全でもリファクタリングの進行を妨げない
  - _Requirements: 6.1, 6.7_
