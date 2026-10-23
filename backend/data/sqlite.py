"""SQLite connections that also close when their transaction scope exits."""
import sqlite3


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def connect(path):
    return sqlite3.connect(path, timeout=30, factory=ClosingConnection)
