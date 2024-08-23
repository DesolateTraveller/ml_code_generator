import streamlit as st

# Set up the page configuration
st.set_page_config(page_title="Login", page_icon="🔒", layout="centered")

# Sample user credentials
USER_CREDENTIALS = {
    "user1": "password1",
    "user2": "password2"
}

# Function to check login
def check_login(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True
    return False

# Main login page
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login to PDF Playground")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
    
    if submit_button:
        if check_login(username, password):
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
else:
    # Your main app content goes here
    st.title("Welcome to PDF Playground")
    st.write("You're logged in as:", st.session_state.username)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()
