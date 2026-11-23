# Changelog

## 1.0.3

### Security

- ネイティブライブラリの読み込みがカレントディレクトリを参照しないよう修正（CWE-427）

### Changed

- MPS バックエンドは、パッケージ同梱の `metalq/lib/libmetalq.dylib` だけを、
  `__file__` を基準にした絶対パスで読み込みます。`native/build/` と
  `~/.metalq/lib/` は探さなくなりました。
- 開発用のビルドを使うときは、環境変数 `METALQ_NATIVE_LIB` に
  `libmetalq.dylib` の絶対パスを指定します。相対パスはエラーになります。
  同じディレクトリに `default.metallib` が必要です。
- ネイティブ側も、`default.metallib` を dylib と同じディレクトリからだけ
  読み込みます。カレントディレクトリ基準の候補は削除しました。
