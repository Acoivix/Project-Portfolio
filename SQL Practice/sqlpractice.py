import csv
import numpy as np
import sqlite3 as sql
from matplotlib import pyplot as plt

def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of -1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    with sql.connect(db_file) as conn:
        # Get the cursor
        cur = conn.cursor() 

        tables = ["MajorInfo", "CourseInfo", "StudentInfo", "StudentGrades"]

        # Drop tables if they exist
        for table in tables:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        
        # Create tables
        cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT)")
        cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT)")
        cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER)")
        cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT)")

        # Insert MajorInfo data
        rows = [(1, "Math"), (2, "Science"), (3, "Writing"), (4, "Art")]
        cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", rows)

        # Insert CourseInfo data
        course_info_rows = [(1, "Calculus"), (2, "English"), (3, "Pottery"), (4, "History")]
        cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", course_info_rows)

        # Insert StudentInfo data
        with open(student_info, "r") as infile:
            student_info_rows = list(csv.reader(infile))
        cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", student_info_rows)

        # Insert StudentGrades data
        with open(student_grades, "r") as infile:
            student_grades_rows = list(csv.reader(infile))
        cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", student_grades_rows)

        # Update StudentInfo tables so that values of -1 in major ID are replaced with NULL
        cur.execute("UPDATE StudentInfo SET MajorID=NULL WHERE MajorID==-1")

        print(student_info_rows)
        print(student_grades_rows)
        
        conn.commit()


def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    with sql.connect(db_file) as conn:
        cur = conn.cursor() 
        # Drop USEarthquakes if it exists
        cur.execute("DROP TABLE IF EXISTS USEarthquakes")
    
        # Create new USEarthquakes
        cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL)")
    
        # Populate table with data from the csv file
        with open("us_earthquakes.csv", "r") as infile:
            rows = list(csv.reader(infile))
        cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows)
        
        # Remove rows from USEarthquakes that have a value of 0 for magnitude
        cur.execute("DELETE FROM USEarthquakes WHERE Magnitude==0")
        
        # Replace 0 values in day, hour, minute, and second columns with NULL values
        cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day==0")
        cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour==0")
        cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute==0")
        cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second==0")
        
        conn.commit()


def grades_format(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()

    # Get tuples of the form (StudentName, CourseName) where that student has an A or A+ grade in that course.
    cur.execute("SELECT SI.StudentName, CI.CourseName "
                "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades AS SG "
                "WHERE SI.StudentID == SG.StudentID AND SG.CourseID == CI.CourseID AND (SG.Grade = 'A' OR SG.Grade = 'A+')")

    # Get the entire set of tuples
    tuples = cur.fetchall()

    # Close the connection
    conn.close()

    return tuples


def create_figure(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()

    # Query the USEarthQuakes Table for the magnitudes of the earthquakes during the 19th century 
    cur.execute("SELECT US.Magnitude "
                "FROM USEarthquakes as US "
                "WHERE US.Year BETWEEN 1800 and 1899")
    
    nineteenth_century_mags = cur.fetchall()
    nineteenth_century_mags = np.ravel(nineteenth_century_mags)     # Use np.ravel() to convert the list of tuples to a 1-D array
    
    # Query the USEarthQuakes Table for the magnitudes of the earthquakes during the 20th century
    cur.execute("SELECT US.Magnitude "
                "FROM USEarthquakes AS US "
                "WHERE US.Year BETWEEN 1900 and 2000")
    
    twentieth_century_mags = cur.fetchall()
    twentieth_century_mags = np.ravel(twentieth_century_mags)

    # Query the USEarthQuakes Table for the average magnitude of all Earthquakes in the database
    cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes;")
    avg = cur.fetchall()

    # Plot figure with subplots of histogram of magnitudes of Earthquakes from each century
    plt.figure(figsize=(12, 8))
    plt.suptitle("Comparison of Magnitudes of Earthquakes in different centuries")

    ax1 = plt.subplot(121)
    ax1.hist(nineteenth_century_mags) 
    ax1.set_title("Magnitudes of Earthquakes from the 19th Century")
    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel("Occurences")

    ax2 = plt.subplot(122)
    ax2.hist(twentieth_century_mags)
    ax2.set_title("Magnitudes of Earthquakes from the 20th Century")
    ax2.set_xlabel("Magnitude")
    ax2.set_ylabel("Occurences")

    plt.show()
    
    # Close the connection
    conn.close()

    # Return average magnitude of all the earthquakes in the database
    avg = np.ravel(avg)
    return avg[0]


if __name__ == "__main__":
    # Test student database and earthquake database
    student_db("students.db")
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM StudentInfo;")
        print([d[0] for d in cur.description])

   
    student_db("students.db")
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM MajorInfo;"):
            print(row)

    
    earthquakes_db("earthquakes.db")
    with sql.connect("earthquakes.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM USEarthquakes;"):
            print(row)

    
    student_db("students.db")
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM StudentInfo;"):
            print(row)

    earthquakes_db("earthquakes.db")
    with sql.connect("earthquakes.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM USEarthquakes;"):
            print(row)
        
    # Test grades_format function
    print(grades_format())

    # Test figure
    avg = create_figure()
    print(avg)