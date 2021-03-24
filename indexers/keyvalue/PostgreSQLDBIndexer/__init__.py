__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable

from jina.executors.indexers.keyvalue import BinaryPbIndexer
from jina.helper import cached_property

if False:
    from ..PostgreSQLDBIndexer.postgresqldbhandler import PostgreSQLDBHandler


class PostgreSQLDBIndexer(BinaryPbIndexer):
    """:class:`PostgreSQLDBIndexer` PostgreSQL based KV Indexer."""

    # TODO user name and password as environment variables
    def __init__(self,
                 hostname: str = '127.0.0.1',
                 port: int = 5432,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 database: str = 'postgres',
                 *args, **kwargs):
        """
        Initialize the PostgreSQLDBIndexer.

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

    def post_init(self):
        """Initialize the PostgreSQLDBHandler inside the Indexer."""
        from .postgresqldbhandler import PostgreSQLDBHandler
        super().post_init()
        self.handler = PostgreSQLDBHandler(hostname=self.hostname,
                                           port=self.port,
                                           username=self.username,
                                           password=self.password,
                                           database=self.database_name)

    def get_add_handler(self) -> 'PostgreSQLDBHandler':
        """Get the handler to PostgreSQL."""
        return self.handler

    def get_create_handler(self) -> 'PostgreSQLDBHandler':
        """Get the handler to PostgreSQL."""
        return self.handler

    def get_query_handler(self) -> 'PostgreSQLDBHandler':
        """Get the handler to PostgreSQL."""
        return self.handler

    def add(self, keys: Iterable[str], values: Iterable[bytes], *args, **kwargs) -> None:
        """Add a Document to PostgreSQL.

        :param keys: the keys by which you will add
        :param values: the values
        :param args: other args
        :param kwargs: other kwargs
        """
        with self.write_handler as PostgreSQL_handler:
            for i, j in zip(keys, values):
                doc = {'_id': i, 'values': j}
                inserted_ids = PostgreSQL_handler.add(documents=[doc])
                if len(inserted_ids) != 1:
                    raise Exception(f'Mismatch in PostgreSQL insert')

    @cached_property
    def query_handler(self) -> 'PostgreSQL_handler':
        """Get the handler to PostgreSQL."""
        return self.get_query_handler()

    def query(self, key: str, *args, **kwargs) -> Optional[bytes]:
        """Query the serialized documents by document id.

        :param key: document id
        :return: serialized document
        """
        with self.query_handler as PostgreSQL_handler:
            result = PostgreSQL_handler.query(key)

        if result:
            return result

    def update(self, keys: Iterable[str], values: Iterable[bytes], *args, **kwargs) -> None:
        """Update the document at the given key in the database.

        :param keys: document ids to update
        :param values: serialized documents
        """
        with self.query_handler as PostgreSQL_handler:
            PostgreSQL_handler.update(keys, values)

    def delete(self, keys: Iterable[str], *args, **kwargs) -> None:
        """Delete documents from the indexer.

        :param keys: document ids to delete
        """
        with self.query_handler as PostgreSQL_handler:
            PostgreSQL_handler.delete(keys)
