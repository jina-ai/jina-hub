from typing import Dict, Optional, Iterable

from jina.logging.logger import JinaLogger


class PostgreSQLDBException(Exception):
    """Any errors raised by PostgreSQL.
    """


class PostgreSQLDBHandler:
    """PostgreSQL Handler to connect to the database and can apply add, update, delete and query.
    PostgreSQL has no access control by default, hence it can be used without username:password.

    The `hstore` data type is a key-value store embedded in PostgreSQL.
    Psycopg can convert Python dict objects to and from hstore structures.
    Only dictionaries with string/unicode keys and values are supported.
    `None` is also allowed as value but not as a key.
    """

    def __init__(self,
                 hostname: str = '127.0.0.1',
                 port: int = 5432,
                 username: str = 'default_name',
                 password: Optional[str] = None,
                 database: str = 'default_db'):
        self.logger = JinaLogger(self.__class__.__name__)
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database


    def __enter__(self):
        return self.connect()

    def connect(self) -> 'PostgreSQLDBHandler':
        """Connect to the database.
        """
        import psycopg2
        from psycopg2 import Error, extras

        try:
            self.connection = psycopg2.connect(user=self.username,
                                            password=self.password,
                                            database=self.database_name,
                                            host=self.hostname,
                                            port=self.port)
            psycopg2.extras.register_hstore(self.connection) #for key-value
            self.cursor = self.connection.cursor()
            self.logger.info('Successfully connected to the database')
        except (Exception, Error) as error:
            print('Error while connecting to PostgreSQL', error)
        finally:
            if self.connection:
                self.connection.close()
                self.cursor.close()
                print('PostgreSQL connection is closed')
        return self

    @property
    def database(self) -> 'Database':
        """ Get database. """
        return self.connection.get_dsn_parameters()

    def add(self, documents: Iterable[Dict]) -> Optional[str]:
        """ Insert the documents into the database.

        :param documents: documents to be inserted
        """
        from psycopg2 import extras, Error
        try:
            table_sql = """
            create table hist_table(
            id text,
            values bytes);
            """
            self.cursor.execute(table_sql)
            sql = """
                INSERT INTO hist_table (id, values) VALUES (%s, ?)"
                """
            self.cursor.execute(sql, documents)
        except (Exception, Error) as error:
            print('Got an error while inserting a document in the db ', error)

    def __exit__(self, *args):
        """ Make sure the connection to the database is closed.
        """
        from psycopg2 import Error
        try:
            self.connection.close()
            self.cursor.close()
            print('PostgreSQL connection is closed')
        except (Exception, Error) as error:
            print('Error while closing: ', error)