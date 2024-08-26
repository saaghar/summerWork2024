#Database builder for ground truth data
import sqlite3

def createDatabase():
    conn = sqlite3.connect('dataBase.db')
    cursor = conn.cursor()


    cursor.execute('''CREATE TABLE IF NOT EXISTS centers_dataset1 (
                        id INTEGER PRIMARY KEY,
                        center_x REAL,
                        center_y REAL)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS radius_dataset1 (
                        id INTEGER PRIMARY KEY,
                        radius REAL)''')
    
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS centers_dataset2 (
                        id INTEGER PRIMARY KEY,
                        center_x REAL,
                        center_y REAL)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS radius_dataset2 (
                        id INTEGER PRIMARY KEY,
                        radius REAL)''')

    conn.commit()
    conn.close()

def insertDatabase(centers, radius, dataset_id):
    try:
        conn = sqlite3.connect('dataBase.db')
        cursor = conn.cursor()

        if dataset_id == 1:
            
            if centers:
                cursor.executemany('''INSERT INTO centers_dataset1 (center_x, center_y)
                                      VALUES (?, ?)''', 
                                      [(center[0], center[1]) for center in centers])

            if radius:
                cursor.executemany('''INSERT INTO radius_dataset1 (radius)
                                      VALUES (?)''', 
                                      [(r,) for r in radius])

        elif dataset_id == 2:
           
            if centers:
                cursor.executemany('''INSERT INTO centers_dataset2 (center_x, center_y)
                                      VALUES (?, ?)''', 
                                      [(center[0], center[1]) for center in centers])

            if radius:
                cursor.executemany('''INSERT INTO radius_dataset2 (radius)
                                      VALUES (?)''', 
                                      [(r,) for r in radius])

        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()




