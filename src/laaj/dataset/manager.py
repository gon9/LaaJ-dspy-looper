"""データセット管理クラス"""

import json
from pathlib import Path

import dspy


class DatasetManager:
    """データセットの読み込み・分割・変換を管理するクラス"""

    def __init__(self, dataset_path: str | Path, input_fields: list[str]):
        """データセットマネージャを初期化

        Args:
            dataset_path: データセットファイルのパス（JSONL形式）
            input_fields: 入力フィールド名のリスト
        """
        self.dataset_path = Path(dataset_path)
        self.input_fields = input_fields
        self._raw_data: list[dict] = []

    def load(self) -> None:
        """データセットファイルを読み込む

        Raises:
            FileNotFoundError: データセットファイルが存在しない場合
            ValueError: JSONL形式が不正な場合
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"データセットファイルが見つかりません: {self.dataset_path}")

        self._raw_data = []
        with open(self.dataset_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._raw_data.append(data)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSONL形式エラー（行 {line_num}）: {e}"
                    ) from e

        if not self._raw_data:
            raise ValueError("データセットが空です")

    def split(
        self, train_ratio: float, val_ratio: float, test_ratio: float
    ) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
        """データセットを訓練/検証/テストに分割

        Args:
            train_ratio: 訓練データの割合
            val_ratio: 検証データの割合
            test_ratio: テストデータの割合

        Returns:
            tuple: (訓練データ, 検証データ, テストデータ)のタプル

        Raises:
            ValueError: データがロードされていない場合
        """
        if not self._raw_data:
            raise ValueError("データセットがロードされていません。load()を先に実行してください")

        total = len(self._raw_data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_data = self._raw_data[:train_size]
        val_data = self._raw_data[train_size : train_size + val_size]
        test_data = self._raw_data[train_size + val_size :]

        train_examples = self._convert_to_examples(train_data)
        val_examples = self._convert_to_examples(val_data)
        test_examples = self._convert_to_examples(test_data)

        return train_examples, val_examples, test_examples

    def _convert_to_examples(self, data: list[dict]) -> list[dspy.Example]:
        """辞書のリストをdspy.Exampleのリストに変換

        Args:
            data: 辞書のリスト

        Returns:
            list[dspy.Example]: dspy.Exampleのリスト
        """
        examples = []
        for item in data:
            example = dspy.Example(**item).with_inputs(*self.input_fields)
            examples.append(example)
        return examples

    def get_all_examples(self) -> list[dspy.Example]:
        """全データをdspy.Exampleのリストとして取得

        Returns:
            list[dspy.Example]: 全データのdspy.Exampleリスト

        Raises:
            ValueError: データがロードされていない場合
        """
        if not self._raw_data:
            raise ValueError("データセットがロードされていません。load()を先に実行してください")

        return self._convert_to_examples(self._raw_data)

    @property
    def size(self) -> int:
        """データセットのサイズを取得

        Returns:
            int: データセットのサイズ
        """
        return len(self._raw_data)
