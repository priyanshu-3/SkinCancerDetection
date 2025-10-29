#!/usr/bin/env python3
"""
Fix NULL created_at timestamps in the analysis table
"""

import sqlite3
from datetime import datetime, timedelta

def fix_null_dates():
    """Update NULL created_at values with reasonable timestamps"""
    
    # Connect to the database
    conn = sqlite3.connect('instance/skin_cancer.db')
    cursor = conn.cursor()
    
    try:
        # Get analyses with NULL created_at
        cursor.execute("""
            SELECT id, patient_name, diagnosis 
            FROM analysis 
            WHERE created_at IS NULL
            ORDER BY id DESC
        """)
        
        null_analyses = cursor.fetchall()
        
        print(f"Found {len(null_analyses)} analyses with NULL created_at:")
        for analysis in null_analyses:
            print(f"  ID: {analysis[0]}, Patient: {analysis[1]}, Diagnosis: {analysis[2]}")
        
        if null_analyses:
            # Update each NULL created_at with a timestamp
            # Use a base date and add some variation based on ID
            base_date = datetime(2025, 10, 27, 10, 0, 0)  # October 27, 2025, 10:00 AM
            
            for i, analysis in enumerate(null_analyses):
                analysis_id = analysis[0]
                # Add some variation: older IDs get earlier timestamps
                hours_offset = (len(null_analyses) - i) * 2  # 2 hours between each
                created_at = base_date - timedelta(hours=hours_offset)
                
                # Update the record
                cursor.execute("""
                    UPDATE analysis 
                    SET created_at = ? 
                    WHERE id = ?
                """, (created_at, analysis_id))
                
                print(f"Updated ID {analysis_id} with timestamp: {created_at}")
            
            # Commit the changes
            conn.commit()
            print(f"\n✅ Successfully updated {len(null_analyses)} records")
        else:
            print("No NULL created_at values found")
        
        # Verify the fix
        cursor.execute("""
            SELECT id, patient_name, created_at 
            FROM analysis 
            WHERE created_at IS NULL
        """)
        
        remaining_nulls = cursor.fetchall()
        if remaining_nulls:
            print(f"⚠️ Warning: {len(remaining_nulls)} records still have NULL created_at")
        else:
            print("✅ All created_at values are now set")
            
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    fix_null_dates()
