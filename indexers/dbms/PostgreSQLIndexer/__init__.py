__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple, Generator
import numpy as np

from jina.executors.indexers.dbms import BaseDBMSIndexer
from jina.executors.indexers.dump import export_dump_streaming

from .postgreshandler import PostgreSQLDBMSHandler


class PostgreSQLDBMSIndexer(BaseDBMSIndexer):
    """:class:`PostgreSQLDBMSIndexer` PostgreSQL based BDMS Indexer.
    Initialize the PostgreSQLDBIndexer.

    :param hostname: hostname of the machine
    :param port: the port
    :param username: the username to authenticate
    :param password: the password to authenticate
    :param database: the database name
    :param table: the table name to use
    :param args: other arguments
    :param kwargs: other keyword arguments
    """

    def __init__(
            self,
            hostname: str = '127.0.0.1',
            port: int = 5432,
            username: str = 'postgres',
            password: str = '123456',
            database: str = 'postgres',
            table: str = 'default_table',
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.table = table

    def _get_generator(self) -> Generator[Tuple[str, np.array, bytes], None, None]:
        with self.handler as handler:
            # always order the dump by id as integer
            handler.cursor.execute(f"SELECT * from {handler.table} ORDER BY ID::int")
            records = handler.cursor.fetchall()
            for rec in records:
                yield rec[0], rec[1], rec[2]

    @property
    def size(self):
        """Obtain the size of the table

        .. # noqa: DAR201
        """
        with self.handler as postgres_handler:
            postgres_handler.cursor.execute(f"SELECT COUNT(*) from {self.handler.table}")
            records = postgres_handler.cursor.fetchall()
            return records[0][0]

    def post_init(self):
        """Initialize the PostgresHandler inside the Indexer."""
        from .postgreshandler import PostgreSQLDBMSHandler
        super().post_init()
        self.handler = PostgreSQLDBMSHandler(
            hostname=self.hostname,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database,
            table=self.table)

    def get_handler(self) -> 'PostgreSQLDBMSHandler':
        """Get the handler to PostgreSQLDBMS."""
        return self.handler

    def get_add_handler(self) -> 'PostgreSQLDBMSHandler':
        """Get the handler to PostgresSQLDBMS."""
        return self.handler

    def get_create_handler(self) -> 'PostgreSQLDBMSHandler':
        """Get the handler to PostgresSQLDBMS."""
        return self.handler

    def get_query_handler(self) -> 'PostgreSQLDBMSHandler':
        """Get the handler to PostgresSQLDBMS."""
        return self.handler

    def add(self, ids, vecs, metas, *args, **kwargs):
        """Add a Document to PostgreSQLDBMS.

        :param ids: List of doc ids to be added
        :param vecs: List of vecs to be added
        :param metas: List of metas of docs to be added
         """
        with self.handler as postgres_handler:
            postgres_handler.add(ids=ids, vecs=vecs, metas=metas)

    def update(self, ids, vecs, metas, *args, **kwargs):
        """Updated document from the database.

        :param ids: Ids of Docs to be updated
        :param vecs: List of vecs to be updated
        :param metas: List of metas of docs to be updated
        """

        with self.handler as postgres_handler:
            postgres_handler.update(ids=ids, vecs=vecs, metas=metas)

    def delete(self, ids, *args, **kwargs):
        """Delete document from the database.

        :param ids: Ids of Document to be removed
        """

        with self.handler as postgres_handler:
            postgres_handler.delete(ids=ids)

    def dump(self, path, shards):
        """Dump the index

        :param path: the path to which to dump
        :param shards: the nr of shards to which to dump
        """
        export_dump_streaming(
            path,
            shards=shards,
            size=self.size,
            data=self._get_generator()
        )
