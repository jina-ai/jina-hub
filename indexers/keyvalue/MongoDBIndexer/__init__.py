"""This package wraps a Key Value Indexer to the MongoDB NoSQL Database."""
from typing import Optional, Iterator

from jina.executors.indexers.keyvalue import BinaryPbIndexer
from jina.helper import cached_property


class MongoDBIndexer(BinaryPbIndexer):
    """:class:`MongoDBIndexer` MongoDB based KV Indexer."""

    def __init__(self, 
                 hostname: str = '127.0.0.1', 
                 port: int = 27017, 
                 username: str = None, 
                 password: str = None, 
                 database: str = 'defaultdb',
                 collection: str = 'defaultcol',
                 *args, **kwargs):
        """
        Initialize the MongoDBIndexer.

        :param hostname: hostname of the machine
        :param port: the port
        :param username: the username to authenticate
        :param password: the password to authenticate
        :param database: the database name
        :param collection: the collection name
        :param args: other arguments
        :param kwargs: other keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.collection_name = collection

    def post_init(self):
        """Initialize the MongoDBHandler inside the Indexer."""
        from .mongodbhandler import MongoDBHandler
        super().post_init()
        self.handler = MongoDBHandler(hostname=self.hostname,
                                      port=self.port,
                                      username=self.username,
                                      password=self.password,
                                      database=self.database_name,
                                      collection=self.collection_name)

    def get_add_handler(self) -> 'MongoDBHandler':
        """Get the handler to MongoDB."""
        return self.handler

    def get_create_handler(self):
        """Get the handler to MongoDB."""
        return self.handler

    def get_query_handler(self):
        """Get the handler to MongoDB."""
        return self.handler

    def add(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs) -> None:
        """Add a Document to MongoDB.

        :param keys: the keys by which you will add
        :param values: the values
        :param args: other args
        :param kwargs: other kwargs
        """
        with self.write_handler as mongo_handler:
            for i, j in zip(keys, values):
                doc = {'_id': i, 'values': j}
                inserted_ids = mongo_handler.insert(documents=[doc])
                if len(inserted_ids) != 1:
                    raise Exception(f'Mismatch in mongo insert')

    @cached_property
    def query_handler(self):
        """Get the handler to MongoDB."""
        return self.get_query_handler()

    def query(self, key: int, *args, **kwargs) -> Optional[bytes]:
        """Query the Mongo db by keys."""
        with self.query_handler as mongo_handler:
            result = mongo_handler.find(key)

        if result:
            return result

    def update(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs) -> None:
        """
        Update the document at the given key in the database.

        **NOTE**: this completely replaces the document with the key.
        """
        with self.query_handler as mongo_handler:
            mongo_handler.update(keys, values)

    def delete(self, keys: Iterator[int], *args, **kwargs) -> None:
        """Delete the documents with those keys."""
        with self.query_handler as mongo_handler:
            mongo_handler.delete(keys)
