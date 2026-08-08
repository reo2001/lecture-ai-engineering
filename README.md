# AI Engineering Practice Portfolio

AIエンジニアリング実践講座で取り組んだ演習成果物をまとめたリポジトリです。LLMアプリケーション、検索・コンテキスト改善、機械学習モデルの学習・評価、データ品質テスト、GitHub ActionsによるCIを、実際にコードを動かしながら学習しました。

> 本リポジトリには講座で提供された教材・ベースコードが含まれます。講義そのものや教材を作成したものではありません。元教材は [matsuolab/lecture-ai-engineering](https://github.com/matsuolab/lecture-ai-engineering) を参照してください。

## Repository Overview

| Directory | Focus | Main artifacts |
| --- | --- | --- |
| [`day1/`](./day1/) | LLMアプリケーション開発 | Streamlit UI、Gemmaチャット、回答評価、SQLite、FastAPI、ngrok |
| [`day3/`](./day3/) | LLM出力の改善 | 講義テキストの前処理、チャンク化、検索、Rerankを扱うJupyter Notebook |
| [`day5/`](./day5/) | MLOps / CI | Titanicモデル学習、MLflow、Kedro、データ検証、pytest、GitHub Actions |
| [`.github/workflows/`](./.github/workflows/) | Continuous Integration | 静的検査、フォーマット確認、データテスト、モデルテスト |

## What I Practiced

### LLM application and evaluation

- Streamlitを使ったチャットUIとフィードバック画面の構成
- Hugging Face TransformersによるGemma 2モデルの推論
- 応答時間、文字数、トークン数などの評価指標表示
- BLEU、TF-IDFコサイン類似度、関連性スコアの計算
- SQLiteへの会話履歴・評価結果の保存と可視化
- FastAPIによるローカルLLMのAPI化とngrok経由の接続実験

### LLM response improvement

- 講義文字起こしデータのクリーニングとチャンク化
- 質問に関連するコンテキストの検索
- Rerankによる取得結果の改善
- GPU条件に応じたL4版・T4版Notebookの比較

### Machine learning and MLOps

- Titanicデータセットを使ったRandom Forestの学習とaccuracy評価
- 前処理、学習、評価、モデル保存をまとめたscikit-learn Pipeline
- MLflowによる実験記録とモデル管理
- Kedroを使った処理パイプラインの構成
- pytestとGreat Expectationsによるデータ型、欠損率、値範囲の検証
- 精度、推論時間、再現性、保存済みモデルの推論を確認するモデルテスト
- Black、Flake8、GitHub Actionsを組み合わせたCI

## My Contributions

コミット履歴から確認できる、`reo2001` による主な追加・変更は次のとおりです。

- `day1/02_streamlit_app/llm.py`, `ui.py`
  - LLM応答の文字数・トークン数を計測し、Streamlit画面へ表示
  - 履歴画面の評価指標表示を拡張
- `day5/演習3/tests/test_saved_model.py`
  - 保存済みTitanicモデルの存在、推論件数、出力ラベルを検証するpytestテストを実装
- `.github/workflows/test.yml`
  - `master` へのpushでCIを起動するトリガーを追加
  - モデル検証をCI対象へ含め、依存関係を専用requirementsへ整理

上記の評価指標をSQLiteへ保存し、履歴・分析画面から確認できるように実装しています。

## Technologies

| Category | Technologies |
| --- | --- |
| Language / Notebook | Python, Jupyter Notebook |
| LLM / NLP | Hugging Face Transformers, Gemma 2, NLTK, Janome |
| Application | Streamlit, FastAPI, Uvicorn, SQLite, ngrok |
| Machine Learning | pandas, NumPy, scikit-learn, Random Forest |
| MLOps | MLflow, Kedro, model serialization |
| Quality | pytest, Great Expectations, Black, Flake8 |
| CI | GitHub Actions |

## Directory Structure

```text
.
├── .github/workflows/test.yml       # Day 5のテスト・静的検査CI
├── day1/
│   ├── 01_streamlit_UI/             # Streamlit UIの基礎
│   ├── 02_streamlit_app/            # LLMチャット、評価、履歴保存
│   ├── 03_FastAPI/                  # ローカルLLM APIとクライアント
│   └── day1_practice.ipynb          # Day 1演習Notebook
├── day3/
│   ├── ai_engineering_03.ipynb      # L4 GPU向け演習
│   ├── ai_engineering_03_T4.ipynb   # T4 GPU向け演習
│   └── data/                         # 検索・改善に使用する講義テキスト
└── day5/
    ├── 演習1/                       # 学習、MLflow、Kedro
    ├── 演習2/                       # データ整合性・フォーマット検査
    ├── 演習3/                       # pytestとCI対象コード
    ├── requirements.txt             # Day 5演習全体の依存関係
    └── requirements-ci.txt          # CI・テスト用の最小依存関係
```

## Setup and Validation

リポジトリを取得し、Day 5のテスト環境を準備します。

```bash
git clone https://github.com/reo2001/lecture-ai-engineering.git
cd lecture-ai-engineering
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r day5/requirements-ci.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r day5/requirements-ci.txt
```

CIと同じ対象をローカルで確認します。

```bash
flake8 day5/演習3 --count --select=E9,F63,F7,F82 --show-source --statistics
black --check day5/演習3
pytest day5/演習3/tests/test_data.py -v
pytest day5/演習3/tests/test_model.py day5/演習3/tests/test_saved_model.py -v
```

Day 1・Day 3の詳細な準備やGPU・Hugging Faceモデルの要件は、各ディレクトリのREADMEを参照してください。

## Notes

- Notebookや講義データは演習内容を確認できるように保持しています。
- Hugging Face・ngrokのトークンはリポジトリへ保存せず、環境変数またはColab / StreamlitのSecretsを使用します。
- `.env_template` に含まれる値はプレースホルダーであり、実際のトークンではありません。
- 学習済み `.pkl` は講義演習の再現用です。信頼できないpickleファイルは任意コード実行の危険があるため読み込まないでください。
- LLM Notebookの実行には、モデルライセンスへの同意と十分なGPUメモリが必要です。
