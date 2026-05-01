import sqlite3, os
os.chdir(r"c:\Users\NITYA\Desktop\Python\SGP")
conn = sqlite3.connect("users.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", [t[0] for t in tables])
try:
    cnt = conn.execute("SELECT COUNT(*) FROM ipo_listings").fetchone()[0]
    print("IPO listings:", cnt)
except Exception as e:
    print("ipo_listings not seeded yet (will seed on first Flask run):", e)
conn.close()
print("DB check complete.")
