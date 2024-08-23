import streamlit as st

# Function to validate login credentials
def validate_login(user_id, password):
    # Replace with your actual validation logic
    valid_user_id = "admin"
    valid_password = "password123"

    return user_id == valid_user_id and password == valid_password

# Function to handle login
def login():
    st.title("Login to PDF Playground")

    # Create two columns for the login page
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <h2>Welcome to PDF Playground</h2>
            <p>An easy-to-use, open-source PDF application to preview and extract content and metadata from PDFs, add or remove passwords, modify, merge, convert, and compress PDFs.</p>
            <p><i>Created by Avijit Chakraborty</i></p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("## Please Login")

        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if validate_login(user_id, password):
                st.session_state.logged_in = True
                st.experimental_rerun()  # Refresh the app to move to the main page
            else:
                st.error("Invalid User ID or Password")

# Function to handle the main app content
def main_app():
    st.title("Welcome to the Main App")
    st.write("This is the main content of the app. You are logged in!")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# Main logic of the app
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:
    main_app()
