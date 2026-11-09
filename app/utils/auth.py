import hashlib
import hmac

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


DEFAULT_USERNAME = "ange.desire"
DEFAULT_PASSWORD_HASH = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
DEFAULT_DISPLAY_NAME = "Ange Désiré Boua"
DEFAULT_ROLE = "Analyste fraude"
PASSWORD_SALT = b"mobilemoney-fraud-pfe"


def _hash_password(password):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        PASSWORD_SALT,
        210_000
    ).hex()


def _load_auth_profile():
    try:
        configured = dict(st.secrets["auth"])
    except (StreamlitSecretNotFoundError, KeyError):
        configured = {}

    username = str(configured.get("username", DEFAULT_USERNAME)).strip()
    display_name = str(
        configured.get("display_name", DEFAULT_DISPLAY_NAME)
    ).strip()
    role = str(configured.get("role", DEFAULT_ROLE)).strip()

    if configured.get("password"):
        password_hash = _hash_password(str(configured["password"]))
    else:
        password_hash = str(
            configured.get("password_hash", DEFAULT_PASSWORD_HASH)
        ).strip()

    return {
        "username": username,
        "display_name": display_name,
        "role": role,
        "password_hash": password_hash
    }


def initialize_session():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("login_error", "")


def authenticate(username, password):
    profile = _load_auth_profile()
    username_matches = hmac.compare_digest(
        username.strip().casefold(),
        profile["username"].casefold()
    )
    password_matches = hmac.compare_digest(
        _hash_password(password),
        profile["password_hash"]
    )

    if not (username_matches and password_matches):
        return None

    return {
        "username": profile["username"],
        "display_name": profile["display_name"],
        "role": profile["role"]
    }


def login_user(profile):
    st.session_state["authenticated"] = True
    st.session_state["auth_user"] = profile
    st.session_state["login_error"] = ""


def logout_user():
    selected_theme = st.session_state.get("ui_theme", "dark")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["ui_theme"] = selected_theme
    st.session_state["authenticated"] = False
    st.session_state["auth_user"] = None
    st.session_state["login_error"] = ""


def is_authenticated():
    return bool(
        st.session_state.get("authenticated")
        and st.session_state.get("auth_user")
    )


def require_authentication():
    initialize_session()
    if is_authenticated():
        return
    st.switch_page("app.py")
    st.stop()
