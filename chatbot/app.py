import streamlit as st
import os
import re
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

from datetime import date, datetime, timedelta
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Set up the LLM model
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

form_data = {}

if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "appointment_prompted" not in st.session_state:
    st.session_state["appointment_prompted"] = False
if "appointment_date" not in st.session_state:
    st.session_state["appointment_date"] = None


# Function to extract text from Documents
def document_text(file_path):
    text = ""
    for file in file_path:
        document_reader = PdfReader(file)
        for page in document_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to generate vector store for text embeddings
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


# Function to set up the conversational chain
def get_conversational_chain():
    system_prompt = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(
        template=system_prompt, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function to check if an appointment is being requested
def checking_appointment(chat):
    list_words = ["book", "booking", "reserve", "reservation", "appointment"]
    for word in list_words:
        if word in chat.lower().split():
            return True


# Function to process user input and return the answer from document context
def user_input(user_question, vector_store):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    st.session_state.chat_history.append(
        {"question": user_question, "answer": response["output_text"]}
    )

    st.write("Reply: ", response["output_text"])


# Function to validate user data
def validate_data(name, eamil, phone):
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
    phone_pattern = r"^\+?[1-9]\d{9,14}$"

    if not re.match(email_pattern, eamil):
        st.error("Invalid email address")
        return False

    if not re.match(phone_pattern, phone):
        st.error("Invalid phone number")
        return False
    return True


# Function to collect user information
def collect_user_info():
    global form_data
    placeholder = st.empty()
    with placeholder.form("user_info_form", clear_on_submit=True):
        st.info(
            "Let’s introduce youeself. Could I have your name, email, & number PLEASE?"
        )
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone Number")

        submitted = st.form_submit_button("Submit")

        if submitted:
            if validate_data(name, email, phone):
                st.session_state["form_submitted"] = True
                form_data = {"name": name, "email": email, "phone": phone}
                placeholder.empty()
            else:
                st.error("Please correct the errors and try again.")


# Function to calculate exact dates based on user input
def calculate_date(date_string, current_date):
    current_date = datetime.strptime(current_date, "%Y-%m-%d")

    # Handle relative days like "three days"
    if "day" in date_string.lower():
        try:
            num_days = int(date_string.split()[0])
            return (current_date + timedelta(days=num_days)).strftime("%Y-%m-%d")
        except:
            return "Unable to calculate date"

    # Handle "next Monday"
    if "next" in date_string.lower():
        day = date_string.lower().split()[-1]
        days = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        target_day = days.index(day)
        days_ahead = target_day - current_date.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (current_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Handle "tomorrow"
    elif "tomorrow" in date_string.lower():
        return (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

    return "Unable to calculate date"


def create_booking_agent(llm):
    # Define a Tool for date calculation
    date_tool = Tool(
        name="DateCalculator",
        func=calculate_date,
        description="Calculates the date based on the input string and current date",
    )

    tools = [date_tool, TavilySearchResults(max_results=1)]

    # Updated Prompt Template with current_date and required variables
    prompt_template: str = """
You are an AI assistant specialized in booking appointments. Your task is to extract a valid date from the user's input and convert it into the format YYYY-MM-DD.

You are expected to handle:
- Relative dates like "tomorrow", "next Monday", "three days from now", "in two weeks".
- Absolute dates like "15th September 2024", "October 10", or "22/10/2024".
- Ambiguous phrases like "later this week", "next weekend", or "early next month" by estimating the most probable date based on the current date.
- If the sentence contains multiple date references, choose the most relevant one based on context.

Today's date: {current_date}

Available tools: {tool_names}

Tool details:
{tools}

You should follow this process:
1. Thought: Analyze the sentence to understand the context and identify the phrase related to the date.
2. Action: Convert the date or phrase into the format YYYY-MM-DD.
3. If it's a relative date, calculate it based on today’s date.
4. If the date is unclear, make an educated guess and document your reasoning.
5. Return the exact date in YYYY-MM-DD format.

Current conversation:
Human: {input}

Use the following format to present your result:
- Thought: Describe how you are understanding the input and processing it.
- Final Answer: The appointment date in YYYY-MM-DD format.

Some examples of how you should process different types of input:
- Input: "I need an appointment for next Monday."
  Thought: The user wants an appointment on the Monday of the next week. 
  Final Answer: [YYYY-MM-DD for next Monday]
  
- Input: "Schedule the meeting for tomorrow."
  Thought: The user is referring to tomorrow.
  Final Answer: [YYYY-MM-DD for tomorrow]

- Input: "Set the reminder for three days from now."
  Thought: The user wants something scheduled three days from today.
  Final Answer: [YYYY-MM-DD for three days later]

- Input: "Book a trip on 22nd October 2024."
  Thought: The user provided a specific date.
  Final Answer: 2024-10-22.

{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template=prompt_template)

    # Construct the ReAct agent by passing the required 'tools' and 'tool_names'
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor


# Function for bookng appointment
def booking_appointment(booking_time):
    agent_executor = create_booking_agent(llm)
    current_date = date.today().strftime("%Y-%m-%d")

    result = agent_executor.invoke(
        {"input": booking_time, "current_date": current_date}
    )

    return result["output"]


# Main function to run the app
def main():
    global form_data
    st.set_page_config("Chat PDF")
    st.header("Chat with Documents using Gemini 💁")

    if not st.session_state["form_submitted"]:
        collect_user_info()

    if st.session_state["form_submitted"] == True:
        if form_data:
            st.success(
                f"Form submitted {form_data['name']}! You can now upload Files and ask questions from the Files"
            )

        file_path = st.file_uploader("Upload your Files", accept_multiple_files=True)

        if file_path:
            with st.spinner("Processing File..."):
                rew_text = document_text(file_path)
                chunks = get_text_chunks(rew_text)
                vector_store = get_vector_store(chunks)
                st.success("File is processed sucessfullt! You can ask questions.")

            st.subheader("Chat History")
            if st.session_state.chat_history:
                for chat in st.session_state.chat_history:
                    st.write(f"**You**: {chat['question']}")
                    st.write(f"**Assistant**: {chat['answer']}")

            user_question = st.text_input("Ask a Question from the Files")
            if user_question:
                if checking_appointment(user_question) is True:
                    appointment_date = booking_appointment(user_question)
                    st.session_state["appointment_date"] = appointment_date
                    form_data["appointment_date"] = appointment_date
                    print(form_data)
                    st.write(
                        f"Your booking has been registered for {appointment_date}, we will call you on your number for confirmation"
                    )
                else:
                    user_input(user_question, vector_store)


if __name__ == "__main__":
    main()
