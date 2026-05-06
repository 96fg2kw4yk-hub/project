import pymysql

from pymysql.cursors import DictCursor
from configuration.config import *
from neo4j import GraphDatabase
import configuration.config

class MysqlReader:
    def __init__(self):
        self.connection = pymysql.connect(**MYSQL_CONFIG)
        self.cursor = self.connection.cursor(DictCursor)
        #查询mysql，读取数据
    def read(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    def close(self):
        self.cursor.close()
        self.connection.close()

class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(**NEO4J_config)

    def write_nodes(self,label:str,properties:list[dict]):
        cypher =f"""
            UNWIND $batch AS item
            MERGE(:{label} {{id:item.id,name:item.name}})
        """
        self.driver.execute_query(cypher,batch=properties)
    def write_relations(self,type:str,start_label,end_label,relations:list[dict]):
        cypher =f"""
                UNWIND $batch AS item
                MATCH (start:{start_label} {{id: item.start_id}}), (end:{end_label} {{id: item.end_id}})
                MERGE(start)-[:{type}]->(end)
        """
        self.driver.execute_query(cypher,batch=relations)

if __name__ == '__main__':
    reader = MysqlReader()
    writer = Neo4jWriter()
    sql = '''
        select id,name
        from
        base_category1
    '''
    category1 = reader.read(sql)
    print(category1)
    writer.write_nodes('Category1', category1)
    sql = """
                  select id, name
                  from 
                    base_category2 
                  """
    category2 = reader.read(sql)
    writer.write_nodes('Category2', category2)
    sql = """
    select id as start_id,
        category1_id as end_id
    from
        base_category2
    """
    relations = reader.read(sql)
    writer.write_relations('Belong', start_label='Category2', end_label='Category1', relations=relations)