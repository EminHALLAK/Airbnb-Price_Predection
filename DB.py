import psycopg2
from psycopg2 import Error


class Database:
    def __init__(self, host, database, user, password, port=5432):
        """Initialize database connection parameters"""
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cursor = self.connection.cursor()
            print("Successfully connected to PostgreSQL")
        except Error as e:
            print(f"Error connecting to PostgreSQL: {e}")

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("PostgreSQL connection closed")

    def execute_query(self, query, params=None):
        """Execute a query"""
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
            return True
        except Error as e:
            print(f"Error executing query: {e}")
            self.connection.rollback()
            return False

    def fetch_all(self, query, params=None):
        """Fetch all results from a query"""
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Error as e:
            print(f"Error fetching data: {e}")
            return None


db = Database(host="localhost", database="airbnb", user="postgres", password="Am2502190.")

