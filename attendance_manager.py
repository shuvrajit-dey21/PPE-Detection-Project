"""
Attendance Management System
Integrated attendance tracking system for PPE monitoring with face recognition
"""

import sqlite3
import pandas as pd
import os
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class AttendanceManager:
    """Comprehensive attendance management system with database integration"""
    
    def __init__(self, db_path: str = "attendance.db"):
        """Initialize the attendance manager
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the attendance database with proper schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create employees table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS employees (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        department TEXT,
                        face_trained BOOLEAN DEFAULT FALSE,
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create attendance records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_id TEXT NOT NULL,
                        date DATE NOT NULL,
                        time_in TIMESTAMP,
                        time_out TIMESTAMP,
                        status TEXT DEFAULT 'Present',
                        camera_location TEXT DEFAULT 'Main Camera',
                        confidence_score REAL,
                        notes TEXT,
                        created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (employee_id) REFERENCES employees (employee_id),
                        UNIQUE(employee_id, date)
                    )
                ''')
                
                # Create attendance sessions table for detailed tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_id TEXT NOT NULL,
                        detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        confidence_score REAL,
                        camera_location TEXT DEFAULT 'Main Camera',
                        session_id TEXT,
                        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_records(date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_employee ON attendance_records(employee_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON attendance_sessions(detection_timestamp)')
                
                conn.commit()
                logging.info("Attendance database initialized successfully")
                
        except Exception as e:
            logging.error(f"Error initializing attendance database: {e}")
            raise
    
    def add_employee(self, employee_id: str, name: str, department: str = None) -> bool:
        """Add a new employee to the database
        
        Args:
            employee_id: Unique employee identifier
            name: Employee name
            department: Employee department (optional)
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO employees (employee_id, name, department, updated_date)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (employee_id, name, department))
                conn.commit()
                logging.info(f"Employee {name} ({employee_id}) added successfully")
                return True
        except Exception as e:
            logging.error(f"Error adding employee: {e}")
            return False
    
    def update_employee_face_status(self, employee_id: str, face_trained: bool = True) -> bool:
        """Update employee's face training status
        
        Args:
            employee_id: Employee identifier
            face_trained: Whether face recognition is trained for this employee
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE employees 
                    SET face_trained = ?, updated_date = CURRENT_TIMESTAMP
                    WHERE employee_id = ?
                ''', (face_trained, employee_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logging.error(f"Error updating employee face status: {e}")
            return False
    
    def get_employee_by_name(self, name: str) -> Optional[Dict]:
        """Get employee information by name
        
        Args:
            name: Employee name
            
        Returns:
            Employee information dictionary or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT employee_id, name, department, face_trained, created_date
                    FROM employees WHERE name = ?
                ''', (name,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'employee_id': row[0],
                        'name': row[1],
                        'department': row[2],
                        'face_trained': bool(row[3]),
                        'created_date': row[4]
                    }
                return None
        except Exception as e:
            logging.error(f"Error getting employee by name: {e}")
            return None
    
    def get_all_employees(self) -> List[Dict]:
        """Get all employees from the database
        
        Returns:
            List of employee dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT employee_id, name, department, face_trained, created_date
                    FROM employees ORDER BY name
                ''')
                rows = cursor.fetchall()
                
                employees = []
                for row in rows:
                    employees.append({
                        'employee_id': row[0],
                        'name': row[1],
                        'department': row[2] or 'Not Specified',
                        'face_trained': bool(row[3]),
                        'created_date': row[4]
                    })
                return employees
        except Exception as e:
            logging.error(f"Error getting all employees: {e}")
            return []

    def record_attendance(self, employee_id: str, confidence_score: float = 0.0,
                         camera_location: str = "Main Camera", session_id: str = None) -> bool:
        """Record attendance for an employee

        Args:
            employee_id: Employee identifier
            confidence_score: Face recognition confidence score
            camera_location: Location of the camera that detected the employee
            session_id: Session identifier for tracking

        Returns:
            True if attendance recorded successfully, False otherwise
        """
        try:
            current_date = date.today()
            current_time = datetime.now()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if attendance already exists for today
                cursor.execute('''
                    SELECT id, time_in FROM attendance_records
                    WHERE employee_id = ? AND date = ?
                ''', (employee_id, current_date))
                existing_record = cursor.fetchone()

                if existing_record:
                    # Update existing record with latest detection time if no time_in set
                    if not existing_record[1]:
                        cursor.execute('''
                            UPDATE attendance_records
                            SET time_in = ?, confidence_score = ?, camera_location = ?
                            WHERE id = ?
                        ''', (current_time, confidence_score, camera_location, existing_record[0]))
                    # If time_in exists, this could be time_out (for future enhancement)
                    logging.info(f"Updated existing attendance record for {employee_id}")
                else:
                    # Create new attendance record
                    cursor.execute('''
                        INSERT INTO attendance_records
                        (employee_id, date, time_in, status, camera_location, confidence_score)
                        VALUES (?, ?, ?, 'Present', ?, ?)
                    ''', (employee_id, current_date, current_time, camera_location, confidence_score))
                    logging.info(f"Created new attendance record for {employee_id}")

                # Always record the detection session for detailed tracking
                cursor.execute('''
                    INSERT INTO attendance_sessions
                    (employee_id, confidence_score, camera_location, session_id)
                    VALUES (?, ?, ?, ?)
                ''', (employee_id, confidence_score, camera_location, session_id))

                conn.commit()
                return True

        except Exception as e:
            logging.error(f"Error recording attendance: {e}")
            return False

    def get_today_attendance(self) -> List[Dict]:
        """Get today's attendance records

        Returns:
            List of today's attendance records
        """
        try:
            current_date = date.today()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT ar.employee_id, e.name, e.department, ar.date, ar.time_in,
                           ar.status, ar.camera_location, ar.confidence_score
                    FROM attendance_records ar
                    JOIN employees e ON ar.employee_id = e.employee_id
                    WHERE ar.date = ?
                    ORDER BY ar.time_in DESC
                ''', (current_date,))
                rows = cursor.fetchall()

                attendance_records = []
                for row in rows:
                    attendance_records.append({
                        'employee_id': row[0],
                        'name': row[1],
                        'department': row[2] or 'Not Specified',
                        'date': row[3],
                        'time_in': row[4],
                        'status': row[5],
                        'camera_location': row[6],
                        'confidence_score': row[7]
                    })
                return attendance_records
        except Exception as e:
            logging.error(f"Error getting today's attendance: {e}")
            return []

    def get_attendance_by_date_range(self, start_date: date, end_date: date) -> List[Dict]:
        """Get attendance records for a date range

        Args:
            start_date: Start date for the range
            end_date: End date for the range

        Returns:
            List of attendance records in the date range
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT ar.employee_id, e.name, e.department, ar.date, ar.time_in,
                           ar.status, ar.camera_location, ar.confidence_score
                    FROM attendance_records ar
                    JOIN employees e ON ar.employee_id = e.employee_id
                    WHERE ar.date BETWEEN ? AND ?
                    ORDER BY ar.date DESC, ar.time_in DESC
                ''', (start_date, end_date))
                rows = cursor.fetchall()

                attendance_records = []
                for row in rows:
                    attendance_records.append({
                        'employee_id': row[0],
                        'name': row[1],
                        'department': row[2] or 'Not Specified',
                        'date': row[3],
                        'time_in': row[4],
                        'status': row[5],
                        'camera_location': row[6],
                        'confidence_score': row[7]
                    })
                return attendance_records
        except Exception as e:
            logging.error(f"Error getting attendance by date range: {e}")
            return []

    def get_attendance_stats(self, target_date: date = None) -> Dict:
        """Get attendance statistics for a specific date

        Args:
            target_date: Date to get stats for (defaults to today)

        Returns:
            Dictionary with attendance statistics
        """
        if target_date is None:
            target_date = date.today()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total employees
                cursor.execute('SELECT COUNT(*) FROM employees')
                total_employees = cursor.fetchone()[0]

                # Present employees for the date
                cursor.execute('''
                    SELECT COUNT(*) FROM attendance_records
                    WHERE date = ? AND status = 'Present'
                ''', (target_date,))
                present_count = cursor.fetchone()[0]

                # Absent employees
                absent_count = total_employees - present_count

                # Attendance rate
                attendance_rate = (present_count / total_employees * 100) if total_employees > 0 else 0

                return {
                    'date': target_date.isoformat(),
                    'total_employees': total_employees,
                    'present_count': present_count,
                    'absent_count': absent_count,
                    'attendance_rate': round(attendance_rate, 2)
                }
        except Exception as e:
            logging.error(f"Error getting attendance stats: {e}")
            return {
                'date': target_date.isoformat(),
                'total_employees': 0,
                'present_count': 0,
                'absent_count': 0,
                'attendance_rate': 0.0
            }

    def export_attendance_to_csv(self, start_date: date = None, end_date: date = None) -> str:
        """Export attendance data to CSV format

        Args:
            start_date: Start date for export (defaults to today)
            end_date: End date for export (defaults to today)

        Returns:
            CSV data as string
        """
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = date.today()

        try:
            attendance_records = self.get_attendance_by_date_range(start_date, end_date)

            if not attendance_records:
                return "No attendance data found for the specified date range."

            # Convert to DataFrame for easy CSV export
            df = pd.DataFrame(attendance_records)

            # Format the time_in column for better readability
            if 'time_in' in df.columns:
                df['time_in'] = pd.to_datetime(df['time_in']).dt.strftime('%H:%M:%S')

            # Reorder columns for better presentation
            column_order = ['employee_id', 'name', 'department', 'date', 'time_in',
                          'status', 'camera_location', 'confidence_score']
            df = df.reindex(columns=[col for col in column_order if col in df.columns])

            return df.to_csv(index=False)

        except Exception as e:
            logging.error(f"Error exporting attendance to CSV: {e}")
            return f"Error exporting data: {str(e)}"

    def export_attendance_to_excel(self, start_date: date = None, end_date: date = None) -> bytes:
        """Export attendance data to Excel format

        Args:
            start_date: Start date for export (defaults to today)
            end_date: End date for export (defaults to today)

        Returns:
            Excel file data as bytes
        """
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = date.today()

        try:
            attendance_records = self.get_attendance_by_date_range(start_date, end_date)

            if not attendance_records:
                # Create empty DataFrame with headers
                df = pd.DataFrame(columns=['employee_id', 'name', 'department', 'date',
                                         'time_in', 'status', 'camera_location', 'confidence_score'])
            else:
                df = pd.DataFrame(attendance_records)

                # Format the time_in column
                if 'time_in' in df.columns:
                    df['time_in'] = pd.to_datetime(df['time_in']).dt.strftime('%H:%M:%S')

            # Create Excel file in memory
            from io import BytesIO
            output = BytesIO()

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Write attendance data
                df.to_excel(writer, sheet_name='Attendance Records', index=False)

                # Add summary sheet
                summary_data = []
                current_date = start_date
                while current_date <= end_date:
                    stats = self.get_attendance_stats(current_date)
                    summary_data.append(stats)
                    current_date += timedelta(days=1)

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Daily Summary', index=False)

                # Format the worksheets
                workbook = writer.book

                # Format attendance sheet
                attendance_sheet = writer.sheets['Attendance Records']
                for column in attendance_sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    attendance_sheet.column_dimensions[column_letter].width = adjusted_width

            output.seek(0)
            return output.getvalue()

        except Exception as e:
            logging.error(f"Error exporting attendance to Excel: {e}")
            return b""

    def reset_daily_attendance(self, target_date: date = None) -> bool:
        """Reset attendance for a specific date

        Args:
            target_date: Date to reset (defaults to today)

        Returns:
            True if reset successfully, False otherwise
        """
        if target_date is None:
            target_date = date.today()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM attendance_records WHERE date = ?
                ''', (target_date,))
                cursor.execute('''
                    DELETE FROM attendance_sessions
                    WHERE DATE(detection_timestamp) = ?
                ''', (target_date,))
                conn.commit()
                logging.info(f"Reset attendance for {target_date}")
                return True
        except Exception as e:
            logging.error(f"Error resetting daily attendance: {e}")
            return False

    def delete_employee(self, employee_id: str) -> bool:
        """Delete an employee and all associated attendance records

        Args:
            employee_id: Employee identifier to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Delete attendance records
                cursor.execute('DELETE FROM attendance_records WHERE employee_id = ?', (employee_id,))
                cursor.execute('DELETE FROM attendance_sessions WHERE employee_id = ?', (employee_id,))

                # Delete employee
                cursor.execute('DELETE FROM employees WHERE employee_id = ?', (employee_id,))

                conn.commit()
                logging.info(f"Deleted employee {employee_id} and all associated records")
                return cursor.rowcount > 0
        except Exception as e:
            logging.error(f"Error deleting employee: {e}")
            return False

    def get_recent_detections(self, limit: int = 10) -> List[Dict]:
        """Get recent face detection sessions

        Args:
            limit: Maximum number of recent detections to return

        Returns:
            List of recent detection sessions
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT s.employee_id, e.name, s.detection_timestamp,
                           s.confidence_score, s.camera_location
                    FROM attendance_sessions s
                    JOIN employees e ON s.employee_id = e.employee_id
                    ORDER BY s.detection_timestamp DESC
                    LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()

                detections = []
                for row in rows:
                    detections.append({
                        'employee_id': row[0],
                        'name': row[1],
                        'detection_timestamp': row[2],
                        'confidence_score': row[3],
                        'camera_location': row[4]
                    })
                return detections
        except Exception as e:
            logging.error(f"Error getting recent detections: {e}")
            return []
