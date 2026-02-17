# gemini.md

## 1. プロジェクト概要
YOLOを用いた、ブラウザベースのリアルタイム物体検知デモアプリ。スマートフォン（背面カメラ）からの入力を主眼に置き、特定のターゲットのみを表示するフィルタリング機能を備える。

## 2. 技術スタック
* **Backend**: FastAPI (Python 3.11+)
* **Frontend**: Alpine.js, Tailwind CSS (CDN利用)
* **Inference**: Ultralytics YOLO (最新の `ultralytics` ライブラリを使用)
* **Infrastructure**: Google Cloud Run (CPU実行)
* **Container**: Docker (Macでのビルド、Cloud Runへのデプロイ)

## 3. ディレクトリ構成
```text
.
├── Dockerfile
├── main.py             # FastAPI backend
├── models/
│   ├── yolo26s.pt      # スマホ、熊、ラップトップ用 (COCO)
│   ├── pen.pt          # カスタムモデル (ID: 0)
│   └── screwdriver.pt  # カスタムモデル (ID: 0)
└── static/
    └── index.html      # Alpine.js frontend
by
```

## 4. モデル・クラスマッピング定義

ターゲット選択に応じて、使用するモデルとフィルタリングするクラスIDを以下の通り定義する。

| 選択肢 (UI) | モデルファイル | 抽出クラスID | 表示ラベル |
| --- | --- | --- | --- |
| スマホ | `yolo26s.pt` | 67 | スマホ |
| 熊 | `yolo26s.pt` | 21 | 熊 |
| ラップトップ | `yolo26s.pt` | 63 | ラップトップ |
| ペン | `pen.pt` | 0 | ペン |
| ドライバー | `screwdriver.pt` | 0 | ドライバー |

## 5. バックエンド仕様 (main.py)

* **Lazy Loading**:
* サーバー起動時にはモデルをロードしない。
* リクエストされたターゲットに応じて、`dict` 形式のキャッシュを確認し、未ロードの場合のみ `YOLO(path)` でメモリに展開する。


* **画像処理**:
* `POST /predict_stream` エンドポイントで画像（Blob）を受信。
* 推論実行後、**選択されたターゲットのみ**にバウンディングボックス（BB）、クラス名、信頼度を描画。
* **重要**: Cloud Runの特性上、ローカルストレージには一切保存しない。
* 描画済み画像は `io.BytesIO` を用いてメモリ内でJPEG変換し、Base64文字列（Data URL形式: `data:image/jpeg;base64,...`）としてレスポンスに含める。


* **レスポンス形式**:
```json
{
  "image": "data:image/jpeg;base64,...",
  "detections": [{"class": "スマホ", "conf": 0.85}]
}

```



## 6. フロントエンド仕様 (index.html)

* **カメラ制御**:
* `getUserMedia` を使用。`facingMode: "environment"` で背面カメラを優先。
* カメラのON/OFF切り替えボタン。


* **UIコンポーネント**:
* ターゲット選択用のセレクトボックス。
* 推論結果表示用の `<img>` タグ。
* モデルロード中のローディングインジケータ（`isLoading` フラグで制御）。
* 追加コード10行程度で、ロード待ちアイコンを表示する。


* **レスポンス対応**:
* サーバーから返却されたBase64文字列を `src` にセットして描画。



## 7. Cloud Run / Docker 仕様

* **Base Image**: `python:3.11-slim`
* **依存関係**: `ultralytics`, `fastapi`, `uvicorn`, `python-multipart`, `opencv-python-headless`, `pillow`
* **実行環境**: CPU実行（メモリ割り当て2GB以上推奨）、同時実行数1。

## 8. 実装時の注意点

* カスタムモデルのクラスIDは **0** とすること。
* 同時実行は想定せず、自分一人がデモを行う前提の最小構成とする。
