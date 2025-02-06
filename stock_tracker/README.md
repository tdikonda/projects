# Stock Tracker

Stock Tracker is a Python script that generates a stock performance report in an HTML table format. It uses Yahoo Finance API (`yfinance`) to fetch the stock data and sends daily email notifications with the report.

## Installation

1. **Install Required Packages**

   You need to install `pandas`, `yfinance`, `natsort`, and other required Python packages. This can be done using pip:

   ```bash
   pip install pandas yfinance natsort
   ```

2. **Set Up Environment Variables**

   The script uses environment variables for sensitive information like email credentials and stock symbols. You need to set these in your system's environment variables or use a `.env` file.

   - `STOCK_TRACKER_EMAIL`: Your email address.
   - `STOCK_TRACKER_PASSWORD`: Your email app-specific password (if required).
   - `STOCK_TRACKER_RECIPIENT`: The recipient email address for the daily report.
   - `STOCK_SYMBOLS`: A comma-separated list of stock symbols to track, e.g., `NVDA,LLY,AAPL`.

## Usage

1. **Run the Script**

   You can run the script using Python:

   ```bash
   python stock_tracker.py
   ```

2. **Example Configuration**

   - Set environment variables in a `.env` file or directly in your system settings:

     ```
     STOCK_TRACKER_EMAIL=your-email@example.com
     STOCK_TRACKER_PASSWORD=your-app-password
     STOCK_TRACKER_RECIPIENT=recipient-email@example.com
     STOCK_SYMBOLS="NVDA,LLY,AAPL,SPY,GOOGL"
     ```

   - Ensure that the SMTP server and port in the script are correctly configured for your email provider.

3. **Output**

   The script will generate an HTML report with stock performance details and send it via email to the recipient specified in the environment variables.

## Example Output

The generated table might look like this:

```
Company Name          | Stock Type | Stock Symbol | Current Price | 52 Week Low | 52 Week High | All Time High (ATH) | ATH Date | % Current Price away from ATH
NVIDIA Corporation    |   EQUITY   |     NVDA   |    $124.65   |   $63.69 |   $153.13    |       $153.13      |01/07/2025|         18.6%
Eli Lilly and Company |   EQUITY   |     LLY   |    $823.23   |   $659.74 |   $972.53    |       $970.92      |08/22/2024|         15.21%
Apple Inc.           |   EQUITY   |     AAPL   |    $237.59   |   $164.08 |   $260.1    |       $260.1      |12/26/2024|         8.65%
SPDR S&P 500 ETF Trust|   ETF    |     SPY   |    $605.04   |   $489.3    |   $610.78    |       $610.78      |01/24/2025|         0.94%
Alphabet Inc.       |   EQUITY   |     GOOGL   |    $200.87   |   $130.67 |   $202.29    |       $202.29      |01/21/2025|         0.7%
```

## Additional Notes

- The script sorts the stocks based on the percentage difference between the current price and the all-time high.
- The email notifications are sent via SMTP. Ensure your email provider allows sending emails through SMTP, and use an app-specific password if necessary.

This script is a simple yet effective way to monitor stock performance and receive daily updates via email.
