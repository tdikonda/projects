# 📈 Agentic Stock Researcher & Advisor

Powered by `Agentic AI`, this application serves as a comprehensive stock research and advisory tool. The `AI Agent` conducts in-depth fundamental and technical analysis, performs risk assessments, aggregates financial analyst recommendations and compiles the latest news to deliver informed stock investment recommendations.

## Demo

![stock_analysis_app_demo](https://github.com/tdikonda/projects/blob/main/ai_agents/stock_analysis_using_crewai/demo/stock_analysis_app_demo.gif)

## Features

*   **Multi-Agent Analysis:** Utilizes CrewAI framework with specialized AI Agents (Researcher, Financial Analyst, News Analyst, Report Writer) to gather and synthesize information.
*   **Comprehensive Analysis:**
    *   **Basic Info:** Company name, sector, industry, market cap, price range.
    *   **Fundamental Analysis:** P/E, P/B, EPS, growth metrics, margins, dividend yield, etc.
    *   **Technical Analysis:** SMAs (50, 200), RSI, MACD, Bollinger Bands with interpretations.
    *   **Risk Assessment:** Volatility, Beta, VaR, Max Drawdown, Sharpe Ratio, Sortino Ratio.
    *   **News & Recommendations:** Collates recent news articles and financial analyst upgrades/downgrades recommendation.
*   **Flexible LLM Support:** Choose between OpenAI, Anthropic or locally running Ollama models.
*   **Customizable Historical Data Analysis:** Select the historical data period for analysis (1 month to maximum available).
*   **Visualizations:** Includes charts for Price/SMAs, RSI, and MACD.
*   **User-Friendly Interface:** Built with Streamlit for easy interaction.

## Inner Workings

Financial data for a stock is fetched from [`Yahoo Finance API`](https://yfinance-python.org/reference/index.html) which is feed into a Large Language Models (LLMs) to generate investment recommendation. The orchestration of tasks and data is done via AI Agents framework [`CrewAI`](https://www.crewai.com/) and presented in a user-friendly [`Streamlit`](https://streamlit.io/) interface.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/tdikonda/projects.git
    cd ai_agents/stock_analysis_using_crewai
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv/Scripts/activate.bat`
    ```

3.  **Install Ollama (if not already installed):**
    *   Ensure you have Ollama installed and running locally. See [Ollama](https://ollama.com/) Installation Guide.
    *   Pull the desired models (e.g. `ollama pull deepseek-r1:14b`). The application will detect available local models.

4.  **Install Dependencies:**
    Ensure you have the required Python packages installed. You can do this using pip:

    ```bash
    pip install crewai crewai-tools matplotlib numpy pandas requests streamlit yfinance
    ```

    OR

    ```bash
    pip install -r requirements.txt
    ```

5.  **API Keys:**
    *   The application will prompt for API keys (OpenAI or Anthropic) in the sidebar if you select those providers. These keys are stored temporarily in memory for the session.
    *   Alternatively, you can set them as environment variables:
        ```bash
        export OPENAI_API_KEY='sk-...'
        export ANTHROPIC_API_KEY='sk-ant-...'
        ```

## Usage

1. **Running the application**

    To run the application, execute the following command:

    ```bash
    streamlit run stock_analysis_app.py
    ```

    This will start a local web server, typically running on `http://localhost:8501`, where you can interact with the app through your web browser.

2. **Interacting with the App**

   - Select a Large Language Model (LLM) provider and select the model.
   - Enter your API key if necessary.
   - Ask questions about stock investment in the `Stock Analysis Query` sidebar text box.