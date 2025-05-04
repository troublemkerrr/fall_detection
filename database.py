import sqlite3
from datetime import datetime

class FallDetectionDB:
    def __init__(self, db_path='fall_detection.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # 事件表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            timestamp DATETIME,
            camera_id INTEGER,
            video_path TEXT,
            status TEXT,
            processed_by TEXT,
            notes TEXT
        )
        ''')
        
        # 关键点数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keypoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            frame_num INTEGER,
            keypoints_data TEXT,
            FOREIGN KEY (event_id) REFERENCES events (event_id)
        )
        ''')
        
        self.conn.commit()
    
    def add_event(self, event_id, timestamp, camera_id, video_path, status="未处理"):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO events (event_id, timestamp, camera_id, video_path, status) VALUES (?, ?, ?, ?, ?)",
            (event_id, timestamp, camera_id, video_path, status)
        )
        self.conn.commit()
    
    def update_event_status(self, event_id, status):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE events SET status=? WHERE event_id=?",
            (status, event_id)
        )
        self.conn.commit()
    
    def add_keypoints(self, event_id, frame_num, keypoints):
        """存储关键点数据"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO keypoints (event_id, frame_num, keypoints_data) VALUES (?, ?, ?)",
            (event_id, frame_num, str(keypoints))
        )
        self.conn.commit()
    
    def get_events(self, limit=50):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        return cursor.fetchall()
    
    def get_event_details(self, event_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events WHERE event_id=?", (event_id,))
        event = cursor.fetchone()
        
        if event:
            cursor.execute("SELECT * FROM keypoints WHERE event_id=? ORDER BY frame_num", (event_id,))
            keypoints = cursor.fetchall()
            return event, keypoints
        return None, None
    
    def close(self):
        self.conn.close()