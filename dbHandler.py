from google.cloud.sql.connector import Connector
import sqlalchemy
from env import INSTANCE_CONNECTION_NAME, DB_USER, DB_PASS, DB_NAME, DB_HOST, DB_PORT


class DBHandler:
    def __init__(self):
        self.is_connected = False

    
    def _getconn(self):
        connector = Connector()
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",  # or "psycopg2"
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
        )
        return conn

    def setupEngine(self):
        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=self._getconn,
            pool_size=5,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=1800,
        )
        self.engine = engine
        self.is_connected = True
    
    def query(self, query, method = "findone"):
        if self.is_connected != True:
            self.setupEngine()

        if method == "findone":
            with self.engine.connect() as connection:
                result = connection.execute(sqlalchemy.text(query))
                return result.fetchone()
        elif method == "findall":
            with self.engine.connect() as connection:
                result = connection.execute(sqlalchemy.text(query))
                return result.fetchall()