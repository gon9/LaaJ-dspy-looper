"""シンプルな分類器のサンプル"""

import dspy


class ClassificationSignature(dspy.Signature):
    """テキストを分類するタスク"""

    text: str = dspy.InputField(desc="分類対象のテキスト")
    label: str = dspy.OutputField(desc="分類ラベル")


class SimpleClassifier(dspy.Module):
    """シンプルな分類器"""

    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(ClassificationSignature)

    def forward(self, text: str):
        """分類を実行

        Args:
            text: 分類対象のテキスト

        Returns:
            分類結果
        """
        return self.classifier(text=text)
