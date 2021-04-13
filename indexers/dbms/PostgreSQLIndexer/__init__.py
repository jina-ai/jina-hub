__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pickle

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
    :param collection: the collection name
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
        self.database_name = database
        self.table = table

    def _get_generator(self):
        self.handler.cursor.execute(f"SELECT * from {self.handler.table} ORDER BY ID")
        record = self.handler.cursor.fetchall()
        for rec in record:
            yield rec[0], rec[1], rec[2]

    def post_init(self):
        """Initialize the PostgresHandler inside the Indexer."""
        from .postgreshandler import PostgreSQLDBMSHandler
        super().post_init()
        self.handler = PostgreSQLDBMSHandler(
            hostname=self.hostname,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database_name,
            table=self.table)

    def get_create_handler(self) -> 'PostgreSQLDBMSHandler':
        """Get the handler to PostgresSQLMDBMS."""
        return self.handler

    def add(self, ids, vecs, metas, *args, **kwargs):
        """Add a Document to PostgresSQLMDBMS.

        :param ids: List of doc ids to be added
        :param vecs: List of vecs to be added
        :param metas: List of metas of docs to be added
        return record: List of Document's id added
         """
        with self.handler as postgres_handler:
            records = postgres_handler.add(ids=ids, vecs=vecs, metas=metas)
        return records

    def update(self, id, vecs, metas, *args, **kwargs):
        """Updated document from the database.

        :param ids: Id of Doc to be updated
        :param vecs: List of vecs to be updated
        :param metas: List of metas of docs to be updated
        :return record: List of Document's id after update
        """

        with self.handler as postgres_handler:
            record = postgres_handler.update(id=id, vecs=vecs, metas=metas)
        return record

    def delete(self, id, *args, **kwargs):
        """Delete document from the database.

        :param id: Id of Document to be removed
        :return record: List of Document's id after deletion
        """

        with self.handler as postgres_handler:
            record = postgres_handler.delete(id=id)
        return record

    def dump(self, path, shards):
        """Dump the index

        :param path: the path to which to dump
        :param shards: the nr of shards to which to dump
        """

        # self.handler.close()
        # noinspection PyPropertyAccess
        # del self.handler
        # self.handler_mutex = False
        # ids = self.query_handler.header.keys()
        export_dump_streaming(
            path,
            shards=shards,
            size=len(list(self._get_generator())),
            data=self._get_generator()
        )
    # self.query_handler.close()
    # self.handler_mutex = False
    # noinspection PyPropertyAccess
    # del self.query_handler
