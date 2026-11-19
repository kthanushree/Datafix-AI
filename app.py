import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import secrets
import hmac
import re
from datetime import datetime

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="DataFix AI",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 DataFix AI")
st.caption("Autonomous Data Quality and Repair Engine")

# ---------------- DATABASE ----------------

def create_database():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt,
        200000
    )


def register_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    salt = secrets.token_bytes(16)
    password_hash = hash_password(password, salt)

    try:
        cursor.execute(
            "INSERT INTO users VALUES (?, ?, ?)",
            (username, salt, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT salt, password_hash FROM users WHERE username=?",
        (username,)
    )

    result = cursor.fetchone()
    conn.close()

    if result is None:
        return False

    salt, stored_hash = result
    entered_hash = hash_password(password, salt)

    return hmac.compare_digest(entered_hash, stored_hash)


create_database()

# ---------------- DETECT ISSUES ----------------

def detect_data_issues(df):
    issues = []

    for column in df.columns:
        if df[column].dtype == "object":

            spaces = df[column].astype(str).str.strip().ne(
                df[column].astype(str)
            )

            if spaces.any():
                issues.append({
                    "Column": column,
                    "Issue": "Extra spaces",
                    "Count": int(spaces.sum())
                })

            if "email" in column.lower():
                invalid = (
                    df[column].notna()
                    & ~df[column].astype(str).str.match(
                        r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
                    )
                )

                if invalid.any():
                    issues.append({
                        "Column": column,
                        "Issue": "Invalid email format",
                        "Count": int(invalid.sum())
                    })

    return pd.DataFrame(
        issues,
        columns=["Column", "Issue", "Count"]
    )


def detect_anomalies(df):
    anomalies = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        values = df[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        unusual = values[
            (values < lower) | (values > upper)
        ]

        for index, value in unusual.items():
            anomalies.append({
                "Column": column,
                "Row": int(index),
                "Value": value,
                "Issue": "Numerical Anomaly"
            })

    return pd.DataFrame(
        anomalies,
        columns=["Column", "Row", "Value", "Issue"]
    )


# ---------------- DIAGNOSE ----------------

def diagnose_issues(df, issues_df, anomalies_df):
    diagnoses = []

    for column in df.columns:
        count = int(df[column].isna().sum())

        if count > 0:
            diagnoses.append({
                "Column": column,
                "Problem": "Missing Values",
                "Likely Cause": "Incomplete data entry",
                "Recommendation": "Review missing values"
            })

    duplicates = int(df.duplicated().sum())

    if duplicates > 0:
        diagnoses.append({
            "Column": "Multiple Columns",
            "Problem": "Duplicate Records",
            "Likely Cause": "Repeated data entry",
            "Recommendation": "Review repeated records"
        })

    for _, issue in issues_df.iterrows():
        diagnoses.append({
            "Column": issue["Column"],
            "Problem": issue["Issue"],
            "Likely Cause": "Possible data entry error",
            "Recommendation": "Verify the original value"
        })

    for _, anomaly in anomalies_df.iterrows():
        diagnoses.append({
            "Column": anomaly["Column"],
            "Problem": "Numerical Anomaly",
            "Likely Cause": "Unusual numerical value",
            "Recommendation": "Verify against original source"
        })

    return pd.DataFrame(diagnoses)


# ---------------- REPAIR ----------------

def clean_dataset(df):
    cleaned = df.copy()
    history = []

    # Remove duplicate records
    duplicate_rows = cleaned[cleaned.duplicated()]

    for index, row in duplicate_rows.iterrows():
        history.append({
            "Row": int(index),
            "Column": "Entire Row",
            "Before": str(row.to_dict()),
            "After": "Duplicate row removed",
            "Reason": "Duplicate record"
        })

    cleaned = cleaned.drop_duplicates()

    # Remove extra spaces
    for column in cleaned.select_dtypes(
        include="object"
    ).columns:

        old_values = cleaned[column].copy()
        new_values = cleaned[column].astype(str).str.strip()

        for index in cleaned.index:
            if old_values.loc[index] != new_values.loc[index]:
                history.append({
                    "Row": int(index),
                    "Column": column,
                    "Before": str(old_values.loc[index]),
                    "After": str(new_values.loc[index]),
                    "Reason": "Removed extra spaces"
                })

        cleaned[column] = new_values.replace("nan", pd.NA)

    # Fill missing values
    for column in cleaned.columns:
        missing_indexes = cleaned.index[
            cleaned[column].isna()
        ]

        if len(missing_indexes) == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned[column]):
            fill_value = cleaned[column].median()
        else:
            mode = cleaned[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"

        for index in missing_indexes:
            history.append({
                "Row": int(index),
                "Column": column,
                "Before": "Missing",
                "After": str(fill_value),
                "Reason": "Filled using median or mode"
            })

        cleaned[column] = cleaned[column].fillna(fill_value)

    return cleaned, pd.DataFrame(history)


# ---------------- VALIDATE ----------------

def validate_dataset(original_df, cleaned_df):
    return pd.DataFrame({
        "Quality Check": [
            "Missing Values",
            "Duplicate Records"
        ],
        "Before Repair": [
            int(original_df.isna().sum().sum()),
            int(original_df.duplicated().sum())
        ],
        "After Repair": [
            int(cleaned_df.isna().sum().sum()),
            int(cleaned_df.duplicated().sum())
        ]
    })


# ---------------- LOGIN / SIGNUP ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    login_tab, signup_tab = st.tabs(
        ["Login", "Create Account"]
    )

    with login_tab:
        st.subheader("Login")

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with signup_tab:
        st.subheader("Create Account")

        new_username = st.text_input(
            "New Username"
        )

        new_password = st.text_input(
            "New Password",
            type="password"
        )

        if st.button("Sign Up"):
            if not new_username or not new_password:
                st.warning("Enter username and password")
            elif register_user(new_username, new_password):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists")

# ---------------- DASHBOARD ----------------

else:

    st.sidebar.success("Logged in")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.header("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.subheader("📋 Original Dataset")
            st.dataframe(df, use_container_width=True)

            # METRICS
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Rows", len(df))
            col2.metric("Columns", len(df.columns))
            col3.metric(
                "Missing Values",
                int(df.isna().sum().sum())
            )
            col4.metric(
                "Duplicates",
                int(df.duplicated().sum())
            )

            # DETECT
            st.header("🔍 DETECT")

            missing = df.isna().sum()
            missing = missing[missing > 0]

            if not missing.empty:
                st.subheader("Missing Values Graph")
                st.bar_chart(missing)
            else:
                st.info("No missing values detected")

            issues_df = detect_data_issues(df)

            st.subheader("Invalid Formats and Inconsistencies")

            if issues_df.empty:
                st.info("No format issues detected")
            else:
                st.dataframe(issues_df)

            anomalies_df = detect_anomalies(df)

            st.subheader("📊 Anomaly Detection")

            if anomalies_df.empty:
                st.info("No numerical anomalies detected")
            else:
                st.dataframe(anomalies_df)

            # DIAGNOSE
            st.header("🧠 DIAGNOSE")

            diagnosis_df = diagnose_issues(
                df,
                issues_df,
                anomalies_df
            )

            if diagnosis_df.empty:
                st.info("No issues found to diagnose")
            else:
                st.dataframe(
                    diagnosis_df,
                    use_container_width=True
                )

            # REPAIR
            st.header("🧹 REPAIR")

            if st.button("Clean Dataset"):

                cleaned_df, history_df = clean_dataset(df)

                st.session_state.cleaned_df = cleaned_df
                st.session_state.history_df = history_df
                st.session_state.original_df = df.copy()

            if "cleaned_df" in st.session_state:

                cleaned_df = st.session_state.cleaned_df
                history_df = st.session_state.history_df
                original_df = st.session_state.original_df

                st.subheader("✨ Cleaned Dataset")

                st.dataframe(
                    cleaned_df,
                    use_container_width=True
                )

                # VALIDATE
                st.header("✅ VALIDATE")

                validation_df = validate_dataset(
                    original_df,
                    cleaned_df
                )

                st.dataframe(
                    validation_df,
                    use_container_width=True
                )

                # EXPLAIN
                st.header("📝 EXPLAIN")

                if history_df.empty:
                    st.info("No modifications were required")
                else:
                    st.dataframe(
                        history_df,
                        use_container_width=True
                    )

                # TRACK
                st.header("📚 TRACK")

                if history_df.empty:
                    st.info("No modification history available")
                else:
                    st.write(
                        "Every recorded change is listed below."
                    )
                    st.dataframe(history_df)

                # QUALITY REPORT
                st.header("📊 QUALITY REPORT")

                before_missing = int(
                    original_df.isna().sum().sum()
                )

                after_missing = int(
                    cleaned_df.isna().sum().sum()
                )

                before_duplicates = int(
                    original_df.duplicated().sum()
                )

                after_duplicates = int(
                    cleaned_df.duplicated().sum()
                )

                st.write(
                    "Missing values resolved:",
                    before_missing - after_missing
                )

                st.write(
                    "Duplicate records removed:",
                    before_duplicates - after_duplicates
                )

                st.write(
                    "Unresolved missing values:",
                    after_missing
                )

                st.write(
                    "Unresolved duplicates:",
                    after_duplicates
                )

                # DOWNLOADS
                st.header("⬇️ DOWNLOAD REPORTS")

                csv_data = cleaned_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Cleaned CSV",
                    data=csv_data,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )

                report_data = validation_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Quality Report",
                    data=report_data,
                    file_name="quality_report.csv",
                    mime="text/csv"
                )

                history_data = history_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Modification History",
                    data=history_data,
                    file_name="modification_history.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error("Unable to process this file.")
            st.exception(e)