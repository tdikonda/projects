'''
Stock tracker python script
Generates a table with columns as below
"Company Name | Stock Type | Stock Symbol | Current Price | 52 Week Low | 52 Week High | All Time High (ATH) | ATH Date | % Current Price away from ATH"

The row values are sorted in descending order based on "% Current Price away from ATH" column value,
which means larger difference between ATH & current price of a stock would make it to the top of the table
'''
import os
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import yfinance as yf
from natsort import natsort_keygen


class StockTracker:
    def __init__(self, email_config):
        """
        Initialize the StockTracker with email configuration and logging setup.

        Args:
            email_config (dict): Dictionary containing email configuration with keys:
                - sender_email: Email address to send from
                - sender_password: App-specific password for email
                - recipient_email: Email address to send notifications to
                - smtp_server: SMTP server address
                - smtp_port: SMTP server port
        """
        self.email_config = email_config

    def create_stock_report(self, stocks_symbol):
        """
        Create a detailed report of all tracked stocks in a pandas DataFrame

        Returns:
            pandas.DataFrame: Table containing stock information
        """
        report_data = []

        for stock_symbol in stocks_symbol:
            try:
                # Get current stock price
                stock = yf.Ticker(stock_symbol)
                historical_data = stock.history(period="max")

                # Find all-time high and its date
                all_time_high = historical_data['High'].max()
                all_time_high_date = historical_data[historical_data['High'] == all_time_high].index[0]

                if stock.info['quoteType'] == 'MUTUALFUND':
                    current_price = stock.history(period="2d")['Close'].iloc[-1]
                else:
                    current_price = stock.history(period="1d")['Close'].iloc[-1]

                # Calculate percentage difference
                percentage_diff = ((all_time_high - current_price) / all_time_high) * 100

                # Add data to report
                report_data.append({
                    'Company Name' : stock.info['longName'],
                    'Stock Type' : stock.info['quoteType'],
                    'Stock Symbol': stock_symbol,
                    'Current Price': f'${round(current_price, 2)}',
                    '52 Week Low' : f'${stock.info['fiftyTwoWeekLow']}',
                    '52 Week High' : f'${stock.info['fiftyTwoWeekHigh']}',
                    'All Time High (ATH)': f'${round(all_time_high, 2)}',
                    'ATH Date': all_time_high_date.strftime('%m/%d/%Y'),
                    '% Current Price away from ATH': f'{round(percentage_diff, 2)}%'
                })

            except Exception as e:
                print(f"Error processing {stock_symbol}: {e}")

        df = pd.DataFrame(report_data)

        # sort values in descending order for '% Current Price away from ATH' column
        df_sorted = df.sort_values(by='% Current Price away from ATH', ascending=False, key=natsort_keygen())

        return df_sorted

    def create_html_report(self, df):
        """
        Create an HTML-formatted report with styling.

        Args:
            df (pandas.DataFrame): Stock data to include in report

        Returns:
            str: HTML-formatted report content
        """
        css_style = """
        <style>
        .styled-table {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            border-collapse: collapse;
            font-size: 14px;
            font-family: Arial, sans-serif;
        }
        .styled-table th,
        .styled-table td {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
            text-align: center;
            background-color: #f2f2f2;
        }
        .styled-table th {
            background-color: #4CAF50;
            font-weight: bold;
        }
        </style>
        """

        current_time = time.strftime('%m/%d/%Y %I:%M:%S %p %Z')
        header_text = f"""
        <h3>Report Time: {current_time}</h3>
        """

        html_table = df.to_html(index=False, classes='styled-table')
        return f"{css_style}\n{header_text}\n{html_table}"

    def _send_email_notification(self, html_content):
        """
        Send email notification with HTML content

        Args:
            html_content (str): HTML-formatted content for email body
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['Subject'] = f'Stock Price Report'
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']

            # Add HTML content
            msg.attach(MIMEText(html_content, 'html'))

            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)

            print("Email notification sent successfully")

        except Exception as e:
            print(f"Failed to send email notification: {e}")

    def run_report(self, stocks_symbol):
        """
        Generate and send the stock report.
        Only runs on weekdays and handles all necessary notifications.
        """
        try:
            print("Starting report generation")

            # Create report
            df = self.create_stock_report(stocks_symbol)
            print(f"Generated report for {len(stocks_symbol)} stocks")

            # Send email
            html_content = self.create_html_report(df)
            self._send_email_notification(html_content)

        except Exception as e:
            print(f"Error in report: {e}")

def main():
    """
    Main function to initialize and run the stock tracker.
    Configures email settings and adds stocks to track.
    """
    email_config = {
        'sender_email': os.environ.get('STOCK_TRACKER_SENDER_EMAIL'),
        'sender_password': os.environ.get('STOCK_TRACKER_SENDER_PASSWORD'),
        'recipient_email': os.environ.get('STOCK_TRACKER_RECIPIENT_EMAIL'),
        'smtp_server': 'smtp.mail.yahoo.com',
        'smtp_port': 587
    }

    tracker = StockTracker(email_config)

    # Get stocks from environment variable
    stocks_str = os.environ.get('STOCK_TRACKER_SYMBOLS')
    print(f'stocks_str is {stocks_str}')

    # Split the comma-separated string into a list
    stocks_symbol = [symbol.strip() for symbol in stocks_str.split(',')]
    print(f'stocks_symbol is {stocks_symbol}')

    # Create stock report
    tracker.create_stock_report(stocks_symbol)

    # Run the report
    tracker.run_report(stocks_symbol)

if __name__ == "__main__":
    main()