import mysql.connector
import json

# cd Database
# pip install mysql-connector-python

# Connect to the database
try:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Sqlinobj?(no)",
        database="chatbot",
    )
    print("Connected to the database!")

    # Write your query
    query = "SHOW TABLES"

    # Execute the query
    cursor = connection.cursor(
        dictionary=True  # Use dictionary=True for JSON compatibility
    )
    cursor.execute(query)

    # Fetch the results
    results = cursor.fetchall()

    # Convert the results to JSON format
    json_data = json.dumps(results, indent=4)  # Pretty-print with indent=4
    print("Data in JSON format:")
    print(json_data)

except mysql.connector.Error as error:
    print("Error:", error)

finally:
    # Close the connection
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("Database connection closed.")
