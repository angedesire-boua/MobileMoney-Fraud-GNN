from html import escape

import streamlit as st

from utils.auth import (
    authenticate,
    initialize_session,
    is_authenticated,
    login_user,
    logout_user
)
from utils.loaders import load_global_style, load_manifest


# Configurer l'application
st.set_page_config(
    page_title="Detection de fraude Mobile Money avec Graph Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto"
)


# Initialiser la session et charger le style global
initialize_session()
load_global_style()
manifest = load_manifest()
threshold = float(manifest["model"]["threshold"])


def switch_theme():
    current_theme = st.session_state.get("ui_theme", "dark")
    st.session_state["ui_theme"] = (
        "light" if current_theme == "dark" else "dark"
    )
    st.rerun()


def render_theme_button(key):
    current_theme = st.session_state.get("ui_theme", "dark")
    target_label = (
        "Activer le mode clair"
        if current_theme == "dark"
        else "Activer le mode sombre"
    )
    target_icon = (
        ":material/light_mode:"
        if current_theme == "dark"
        else ":material/dark_mode:"
    )
    if st.button(
        target_label,
        icon=target_icon,
        key=key,
        help=target_label,
        width="stretch"
    ):
        switch_theme()


def render_login():
    with st.container(key="login_page"):
        header_col, theme_col = st.columns(
            [8, 0.42],
            vertical_alignment="center"
        )
        with header_col:
            st.markdown(
                (
                    '<div class="login-topline">'
                    '<span class="login-logo">MM</span>'
                    '<span>Mobile Money Fraud Detection</span>'
                    '</div>'
                ),
                unsafe_allow_html=True
            )
        with theme_col:
            render_theme_button("login_theme_switch")

        intro_col, form_col = st.columns([1.08, 0.92], gap="small")

        with intro_col:
            with st.container(key="login_intro"):
                st.markdown(
                    (
                        '<div class="login-intro-content">'
                        '<div class="login-kicker">Console antifraude</div>'
                        '<h1>Renseignements sur la fraude Mobile Money</h1>'
                        '<p>Investigation, monitoring et détection sur les '
                        'transactions PaySim.</p>'
                        '<div class="login-model-strip">'
                        '<span><i></i>LightGBM Core</span>'
                        '<span>PaySim · Test</span>'
                        f'<span>Seuil {threshold:.4f}</span>'
                        '</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True
                )

        with form_col:
            with st.container(border=True, key="login_form_panel"):
                st.markdown(
                    (
                        '<div class="login-form-heading">'
                        '<div class="login-avatar">AD</div>'
                        '<div><h2>Connexion analyste</h2>'
                        '<p>Accédez à la console de surveillance.</p></div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True
                )

                with st.form("login_form", border=False):
                    username = st.text_input(
                        "Identifiant",
                        placeholder="ange.desire",
                        autocomplete="username"
                    )
                    password = st.text_input(
                        "Mot de passe",
                        type="password",
                        autocomplete="current-password"
                    )
                    submitted = st.form_submit_button(
                        "Se connecter",
                        type="primary",
                        icon=":material/login:",
                        width="stretch"
                    )

                if submitted:
                    profile = authenticate(username, password)
                    if profile is None:
                        st.session_state["login_error"] = (
                            "Identifiant ou mot de passe incorrect."
                        )
                    else:
                        login_user(profile)
                        st.rerun()

                if st.session_state.get("login_error"):
                    st.error(st.session_state["login_error"])

                st.caption(
                    "Accès local de démonstration. Les identifiants peuvent "
                    "être remplacés dans les secrets Streamlit."
                )

        st.markdown(
            (
                '<div class="login-footer">'
                'Données PaySim · protocole Validation/Test préservé · '
                'aucune sélection sur Test'
                '</div>'
            ),
            unsafe_allow_html=True
        )


def render_application_topbar(user):
    display_name = escape(str(user["display_name"]))
    role = escape(str(user["role"]))
    initials = "".join(
        part[0] for part in str(user["display_name"]).split()[:2]
    ).upper()

    with st.container(key="app_topbar"):
        product_col, context_col, theme_col, user_col, logout_col = st.columns(
            [2.25, 2.1, 0.28, 1.1, 0.28],
            gap="small",
            vertical_alignment="center"
        )

        with product_col:
            st.markdown(
                (
                    '<div class="topbar-product">'
                    '<span class="topbar-mark">MM</span>'
                    '<span>Détection de la fraude à l’argent mobile</span>'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

        with context_col:
            st.markdown(
                (
                    '<div class="topbar-context">'
                    '<span class="topbar-chip topbar-chip--live">'
                    '<span class="status-dot"></span>LightGBM Core</span>'
                    '<span class="topbar-chip">PaySim · Test</span>'
                    f'<span class="topbar-chip">Seuil {threshold:.4f}</span>'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

        with theme_col:
            render_theme_button("theme_switch")

        with user_col:
            st.markdown(
                (
                    '<div class="user-chip">'
                    f'<span class="user-avatar">{escape(initials)}</span>'
                    '<span class="user-copy">'
                    f'<strong>{display_name}</strong>'
                    f'<small>{role}</small>'
                    '</span>'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

        with logout_col:
            if st.button(
                "Se déconnecter",
                icon=":material/logout:",
                key="logout_button",
                help="Se déconnecter",
                width="stretch"
            ):
                logout_user()
                st.rerun()


# Enregistrer les routes avant l'authentification évite le message
# "Page not found" lors d'un accès direct à une page protégée.
dashboard_page = st.Page(
    "pages/1_Tableau_de_bord.py",
    title="Tableau de bord",
    icon=":material/dashboard:",
    default=True
)
detection_page = st.Page(
    "pages/2_Detection.py",
    title="Détection",
    icon=":material/radar:"
)
alerts_page = st.Page(
    "pages/3_Alertes.py",
    title="Alertes",
    icon=":material/notifications_active:"
)
investigation_page = st.Page(
    "pages/4_Enquete.py",
    title="Enquête",
    icon=":material/manage_search:"
)
research_page = st.Page(
    "pages/5_Recherche.py",
    title="Recherche",
    icon=":material/biotech:"
)

application_pages = [
    dashboard_page,
    detection_page,
    alerts_page,
    investigation_page,
    research_page
]
pages = {"Application": application_pages}

# La navigation reste cachée afin de contrôler précisément l'ordre et le
# dessin de la sidebar, tout en enregistrant toutes les routes dès le départ.
navigation = st.navigation(pages, position="hidden")


if not is_authenticated():
    render_login()
    st.stop()


# Identité visuelle et navigation
with st.sidebar:
    st.markdown(
        (
            '<div class="sidebar-brand">'
            '<div class="brand-icon">MM</div>'
            '<div class="brand-text">'
            '<div class="brand-title">RENSEIGNEMENTS SUR LA FRAUDE</div>'
            '<div class="brand-subtitle">Contrôle de l’argent mobile</div>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-section-label">Application</div>',
        unsafe_allow_html=True
    )
    for index, page in enumerate(application_pages):
        state = "active" if navigation.title == page.title else "item"
        with st.container(key=f"sidebar_nav_{state}_{index}"):
            st.page_link(
                page,
                label=page.title,
                icon=page.icon,
                width="stretch"
            )

    with st.container(key="sidebar_footer"):
        st.markdown(
            (
                '<div class="sidebar-footer">'
                '<div class="sidebar-footer-label">MOTEUR DE DÉTECTION</div>'
                '<div class="sidebar-model">'
                '<span class="status-dot"></span>'
                'LightGBM Core'
                '</div>'
                '<div class="sidebar-footer-text">'
                f'Seuil Validation · {threshold:.6f}'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True
        )


# Barre de contexte commune
render_application_topbar(st.session_state["auth_user"])


# Exécuter la page
navigation.run()
