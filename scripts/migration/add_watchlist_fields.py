"""
Database migration: Add watchlist and guard verification fields
Adds criminal/wanted person detection capabilities
"""

import sqlite3
import os
from datetime import datetime

# Database path
DB_PATH = "face_recognition.db"


def migrate():
    """Add watchlist fields to persons and alerts tables"""

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return False

    print(f"🔄 Starting migration on {DB_PATH}...")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Backup database first
        backup_path = f"face_recognition_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        print(f"📦 Creating backup: {backup_path}")
        os.system(f"cp {DB_PATH} {backup_path}")

        print("\n🔨 Adding columns to 'persons' table...")

        # Add watchlist fields to persons table
        try:
            cursor.execute("ALTER TABLE persons ADD COLUMN watchlist_status TEXT DEFAULT 'none' NOT NULL")
            print("  ✅ Added watchlist_status")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("  ⚠️  watchlist_status already exists")
            else:
                raise

        try:
            cursor.execute("ALTER TABLE persons ADD COLUMN threat_level TEXT DEFAULT 'none' NOT NULL")
            print("  ✅ Added threat_level")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("  ⚠️  threat_level already exists")
            else:
                raise

        try:
            cursor.execute("ALTER TABLE persons ADD COLUMN criminal_notes TEXT")
            print("  ✅ Added criminal_notes")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("  ⚠️  criminal_notes already exists")
            else:
                raise

        try:
            cursor.execute("ALTER TABLE persons ADD COLUMN added_to_watchlist_at TIMESTAMP")
            print("  ✅ Added added_to_watchlist_at")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("  ⚠️  added_to_watchlist_at already exists")
            else:
                raise

        try:
            cursor.execute("ALTER TABLE persons ADD COLUMN watchlist_expires_at TIMESTAMP")
            print("  ✅ Added watchlist_expires_at")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e):
                print("  ⚠️  watchlist_expires_at already exists")
            else:
                raise

        print("\n🔨 Adding columns to 'alerts' table...")

        # Add guard verification fields to alerts table
        guard_fields = [
            ("guard_verified", "INTEGER DEFAULT 0 NOT NULL"),
            ("guard_action", "TEXT"),
            ("guard_verified_by", "TEXT"),
            ("guard_verified_at", "TIMESTAMP"),
            ("action_notes", "TEXT"),
            ("threat_level", "TEXT"),
            ("watchlist_status", "TEXT"),
        ]

        for field_name, field_type in guard_fields:
            try:
                cursor.execute(f"ALTER TABLE alerts ADD COLUMN {field_name} {field_type}")
                print(f"  ✅ Added {field_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e):
                    print(f"  ⚠️  {field_name} already exists")
                else:
                    raise

        # Create indexes for new columns
        print("\n🔨 Creating indexes...")

        indexes = [
            ("idx_persons_watchlist", "persons", "watchlist_status"),
            ("idx_persons_threat", "persons", "threat_level"),
            ("idx_alerts_guard_verified", "alerts", "guard_verified"),
            ("idx_alerts_guard_action", "alerts", "guard_action"),
            ("idx_alerts_threat", "alerts", "threat_level"),
            ("idx_alerts_watchlist", "alerts", "watchlist_status"),
        ]

        for idx_name, table, column in indexes:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
                print(f"  ✅ Index created: {idx_name}")
            except sqlite3.OperationalError as e:
                print(f"  ⚠️  Index {idx_name}: {e}")

        conn.commit()
        print("\n✅ Migration completed successfully!")
        print(f"📦 Backup saved: {backup_path}")

        # Show summary
        cursor.execute("PRAGMA table_info(persons)")
        persons_cols = cursor.fetchall()
        cursor.execute("PRAGMA table_info(alerts)")
        alerts_cols = cursor.fetchall()

        print(f"\n📊 Summary:")
        print(f"  - persons table: {len(persons_cols)} columns")
        print(f"  - alerts table: {len(alerts_cols)} columns")

        conn.close()
        return True

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.rollback()
        conn.close()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: Watchlist & Guard Verification")
    print("=" * 60)

    success = migrate()

    if success:
        print("\n✅ Migration complete! Restart your application.")
    else:
        print("\n❌ Migration failed. Check errors above.")

    print("=" * 60)
