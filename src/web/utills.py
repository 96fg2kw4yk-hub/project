from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

from configuration.config import *

class IndexUtil:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_config["uri"],
            username=NEO4J_config["auth"][0],
            password=NEO4J_config["auth"][1],
            database="a262abd8"
        )#连接neo4j

        self.embedding_model = HuggingFaceEmbeddings(
            model_name='BAAI/bge-base-zh-v1.5',
            encode_kwargs={'normalize_embeddings': True},
        )#嵌入768维中文文本嵌入模型

    # 创建全文索引，传入索引名称，节点标签，属性
    def create_fulltext_index(self, index_name, label, property):
        cypher = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON EACH [n.{property}]
        """
        self.graph.query(cypher)
#创建向量索引
    # 创建向量索引，需要传入生成向量的“源属性”，以及嵌入向量属性
    def create_vector_index(self, index_name, label, source_property, embedding_property):
        # 生成嵌入向量，并添加到节点属性中
        embedding_dim = self._add_embedding(label, source_property, embedding_property)

        cypher = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON n.{embedding_property}
            OPTIONS {{ indexConfig: {{
             `vector.dimensions`: {embedding_dim},
             `vector.similarity_function`: 'cosine'
            }} 
            }} 
        """
        self.graph.query(cypher)

    # 内部函数：生成嵌入向量，并添加到节点属性中，返回向量维度
    def _add_embedding(self, label, source_property, embedding_property):
        # 1. 查询所有节点对应的源属性值，作为模型的输入；还需要查出节点id
        cypher = f"""
            MATCH (n:{label})
            RETURN n.{source_property} AS text, id(n) AS id
        """
        results = self.graph.query(cypher)

        # 2. 获取查询结果中的文本内容
        docs = [ result['text'] for result in results ]

        # 3. 调用嵌入模型，得到嵌入向量
        embeddings = self.embedding_model.embed_documents(docs)

        # 4. 将id和嵌入向量组合成字典形式
        batch = []
        for result, embedding in zip(results, embeddings):
            item = {'id': result['id'], 'embedding': embedding}
            batch.append(item)

        # 5. 执行cypher，按id查节点，写入新的嵌入向量属性
        cypher = f"""
            UNWIND $batch AS item
            MATCH (n:{label})
            WHERE id(n) = item.id
            SET n.{embedding_property} = item.embedding
        """
        self.graph.query(cypher, params={"batch": batch})

        return len(embeddings[0])
if __name__ == '__main__':
    index = IndexUtil()
    index.create_fulltext_index("trademark_fulltext_index", "Trademark", "name")
    index.create_vector_index("trademark_vector_index", "Trademark", "name", "embedding")
    index.create_fulltext_index('spu_fulltext_index', 'SPU', 'name')
    index.create_vector_index('spu_vector_index', 'SPU', 'name', 'embedding')
    index.create_fulltext_index('sku_fulltext_index', 'SKU', 'name')
    index.create_vector_index('sku_vector_index', 'SKU', 'name', 'embedding')

    index.create_fulltext_index('category1_fulltext_index', 'Category1', 'name')
    index.create_vector_index('category1_vector_index', 'Category1', 'name', 'embedding')
    index.create_fulltext_index('category2_fulltext_index', 'Category2', 'name')
    index.create_vector_index('category2_vector_index', 'Category2', 'name', 'embedding')
    index.create_fulltext_index('category3_fulltext_index', 'Category3', 'name')
    index.create_vector_index('category3_vector_index', 'Category3', 'name', 'embedding')