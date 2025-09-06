import requests
from typing import List


class SiliconReranker:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def __call__(
        self,
        documents: List[str],
        query: str,
        top_n: int = 4,
        return_documents: bool = True,
    ) -> List[str]:
        
        url = f"{self.base_url}/rerank"

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }


        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.post(url, json=payload, headers=headers)
        response = response.json()

        """
        {
            "id": "<string>",
            "results": [
                {
                "document": {
                    "text": "<string>"
                },
                "index": 123,
                "relevance_score": 123
                }
            ],
            "tokens": {
                "input_tokens": 123,
                "output_tokens": 123
            }
        } 
        """

        results = []

        for result in response["results"]:
            results.append((result["document"]["text"], result["relevance_score"]))

        return results