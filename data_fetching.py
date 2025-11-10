import pandas as pd
import psycopg2
from psycopg2 import Error


def fetch_listing_details():
    """
    Fetch data from listing_details table and store it in a DataFrame
    """
    global connection
    try:
        # Connect to PostgreSQL database
        connection = psycopg2.connect(
            user="postgres",
            password="Am2502190.",  # Update with your actual password
            host="localhost",
            port="5432",
            database="airbnb"
        )

        # Create a cursor object
        cursor = connection.cursor()

        # SQL query to fetch all data from listing_details table
        query = "SELECT * FROM listings_details;"

        # Execute the query
        cursor.execute(query)

        # Fetch all rows
        rows = cursor.fetchall()

        # Get column names
        column_names = [desc[0] for desc in cursor.description]

        # Create DataFrame
        df = pd.DataFrame(rows, columns=column_names)




        return df

    except (Exception, Error) as error:
        print(f"Error while connecting to PostgreSQL: {error}")
        return None

    finally:
            cursor.close()
            connection.close()
            print("\nPostgreSQL connection is closed")


if __name__ == "__main__":
    # Fetch the data
    listing_df = fetch_listing_details()


