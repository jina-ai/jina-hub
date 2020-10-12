from typing import Optional, Iterator

from jina.proto import jina_pb2
from jina.logging.logger import JinaLogger
from jina.helper import cached_property
from jina.executors.indexers.keyvalue import BinaryPbIndexer


class MongoDBIndexer(BinaryPbIndexer):
    """
    :class:`MongoDBIndexer` MongoDB based KV Indexer.
    """

    def __init__(self, 
                 hostname: str = '127.0.0.1', 
                 port: int = 27017, 
                 username: str = None, 
                 password: str = None, 
                 database: str = 'defaultdb',
                 collection: str = 'defaultcol',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.collection_name = collection
        
    def post_init(self):
        from .mongodbhandler import MongoDBHandler
        super().post_init()
        self.handler = MongoDBHandler(hostname=self.hostname,
                                      port=self.port,
                                      username=self.username,
                                      password=self.password,
                                      database=self.database_name,
                                      collection=self.collection_name)
    
    def get_add_handler(self):
        return self.handler

    def get_create_handler(self):
        return self.handler

    def get_query_handler(self):
        return self.handler
    
    def add(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        mongo_docs = [{'_id': i, 'values': j} for i, j in zip(keys, values)]
        
        with self.write_handler as mongo_handler:
            inserted_ids = mongo_handler.insert(documents=mongo_docs)
        
        if inserted_ids and len(inserted_ids) != len(keys):
            self.logger.error(f'Mismatch in mongo insert')
    
    @cached_property
    def query_handler(self):
        return self.get_query_handler()

    def query(self, key: int, *args, **kwargs) -> Optional[bytes]:
        with self.query_handler as mongo_handler:
            result = mongo_handler.find({'_id': key})
        
        if result:
            return result
