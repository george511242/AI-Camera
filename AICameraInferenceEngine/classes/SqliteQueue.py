import sqlite3, os
from typing import Union
from datetime import datetime, timedelta
from pathlib import Path
class SqliteQueue:
    def __init__(self, db_path=':memory:', maxlen: Union[int, None] = 15, max_days: Union[int, None] = None):
        if not db_path == ':memory:':
            os.makedirs(str(Path(db_path).parent), exist_ok=True)
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.maxlen = maxlen
        self.max_days = max_days
        self.create_tables()

    def create_tables(self):
        cursor = self.connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS list_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT,
                value TEXT,
                timestamp TEXT
            )
        ''')
        self.connection.commit()

    def llen(self, key):
        cursor = self.connection.cursor()
        cursor.execute('SELECT COUNT(*) FROM list_data WHERE key = ?', (key,))
        result = cursor.fetchone()
        return result[0] if result else 0

    def lpush(self, key, value):
        # Check if maxlen is exceeded
        if self.maxlen is not None and self.llen(key) >= self.maxlen:
            self.cleanup(key)

        timestamp = datetime.now().isoformat()
        cursor = self.connection.cursor()
        cursor.execute('INSERT INTO list_data (key, value, timestamp) VALUES (?, ?, ?)', (key, value, timestamp))
        self.connection.commit()

    def rpop(self, key):
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT value FROM list_data
            WHERE key = ?
            ORDER BY id DESC
            LIMIT 1
        ''', (key,))
        result = cursor.fetchone()

        if result:
            value = result[0]
            cursor.execute('''
                DELETE FROM list_data
                WHERE id = (
                    SELECT id FROM list_data
                    WHERE key = ?
                    ORDER BY id DESC
                    LIMIT 1
                )
            ''', (key,))
            self.connection.commit()
            return value
        else:
            return None

    def cleanup(self, key):
        # Cleanup records exceeding max_days
        if self.max_days is not None:
            expire_time = datetime.now() - timedelta(days=self.max_days)
            cursor = self.connection.cursor()
            cursor.execute('DELETE FROM list_data WHERE key = ? AND timestamp < ?', (key, expire_time.isoformat()))
            self.connection.commit()

        # Cleanup records exceeding maxlen
        if self.llen(key) > self.maxlen:
            clean_len = self.llen(key) - self.maxlen
            cursor = self.connection.cursor()
            cursor.execute('''
                DELETE FROM list_data
                WHERE id = (
                    SELECT id FROM list_data
                    WHERE key = ?
                    ORDER BY id ASC
                    LIMIT ?
                )
            ''', (key, clean_len))
            self.connection.commit()

    def multipleRpop(self, key, n=1):
        self.cleanup(key)
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT value FROM list_data
            WHERE key = ?
            ORDER BY id DESC
            LIMIT ?
        ''', (key, n))
        results = cursor.fetchall()

        values = []
        for result in results:
            print_data = result
            values.append(result[0])

        if values:
            # Delete the rows corresponding to the popped items
            cursor.execute('''
                DELETE FROM list_data
                WHERE id IN (
                    SELECT id FROM list_data
                    WHERE key = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
            ''', (key, n))
            self.connection.commit()

        return values

    def ping(self):
        return True
    
    def __del__(self):
        self.connection.close()