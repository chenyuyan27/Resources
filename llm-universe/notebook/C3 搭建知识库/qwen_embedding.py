from typing import List
from langchain_core.embeddings import Embeddings

class QwenEmbeddings(Embeddings):
    """`qwen Embeddings` embedding models."""
    def __init__(self):
        """
        实例化qwen为values["client"]

        Args:

            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:

            values (Dict): 包含配置信息的字典。如果环境中有qwen库，则将返回实例化的qwen类；否则将报错 'ModuleNotFoundError: No module named 'qwen''.
        """
        from openai import OpenAI
        client = OpenAI(
        api_key="sk-oxedcjwsmifzqphblncyjkxqlnjeocthxdouevaptgonrrhs",
        base_url="https://api.siliconflow.cn/v1"
    )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """

        result = []
        for i in range(0, len(texts), 64):
            embeddings = self.client.embeddings.create(
                model="Qwen/Qwen3-Embedding-0.6B",
                input=texts[i:i+64]
            )
            result.extend([embeddings.embedding for embeddings in embeddings.data])
        return result
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """

        return self.embed_documents([text])[0]