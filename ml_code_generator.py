import streamlit as st

# Function to validate login credentials
def validate_login(user_id, password):
    # Replace with your actual validation logic
    valid_user_id = "admin"
    valid_password = "password123"
    return user_id == valid_user_id and password == valid_password

# Function to display the login page
def login_page():
    st.title("Login to Access the App")
    
    # Create two columns: one for the app details, the other for the login form
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <h2>Welcome to the App</h2>
            <p>This app requires login to access the content. Please enter your credentials to proceed.</p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("## Please Login")

        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if validate_login(user_id, password):
                st.session_state.logged_in = True
                st.experimental_rerun()
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
