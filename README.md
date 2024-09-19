# Document Query & Appointment Booking Chatbot

This project is a demo that allows users to query content from uploaded PDF documents and simplifies the process of booking appointments through conversational interaction. It is integrated with a form that the user must fill out before using the document query and appointment booking features.

### Project Overview

- **Query Document Content**: Users can upload PDF documents and ask questions about their content.
- **Collect User Information**: Collects user details (Name, Phone Number, Email).
- **Book Appointments**: Provides appointment booking through a conversational form integrated with agent tools.
- **Date Extraction**: Extracts dates from user queries (e.g., "Next Monday") and converts them to standard formats (YYYY-MM-DD).
- **Input Validation**: Validates user inputs, including phone numbers and email addresses.

### Tools and Technologies

- **LangChain**: Connects the language model (LLM) with document storage to enable document querying.
- **Google Gemini**: Provides natural language understanding and interaction with users.
- **Agent Tools**: Handles appointment booking and date processing.
- **Regex**: Used for input validation (email, phone number).
- **Streamlit**: Used for building an interactive web-based user interface (Front-end).

### How to Run the Project

1. **Create a virtual environment**:
    ```bash
    conda create -p venv python=3.10
    ```

2. **Activate the environment**:
    ```bash
    conda activate venv\
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Navigate to the project directory**:
    ```bash
    cd chatbot
    ```

5. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

### Future Enhancements

- Improve the **conversation form** to be more interactive and user-friendly.
- **Support multiple languages** to make the chatbot accessible to a broader audience.
- Enhance the **Streamlit application** for a better user interface and performance.
- Implement **notifications** for booking confirmations via email or phone number using the information collected.

