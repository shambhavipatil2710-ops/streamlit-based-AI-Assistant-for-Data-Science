import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import warnings
import tempfile
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings('ignore')

# Optional wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# Ensure NLTK resources (graceful)
def ensure_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name)
            except Exception:
                # cannot download in this environment
                pass

ensure_nltk_resources()

st.set_page_config(page_title="CSV Dashboard — Chatbot & Report", layout="wide", initial_sidebar_state="expanded")
st.title("CSV Dashboard — Chatbot & Abstract Report ✨")

# Helper: load dataframe
@st.cache_data
def load_dataframe(file) -> pd.DataFrame:
    return pd.read_csv(file)

# Helper: best-match column finder
def find_column_by_name(query_col, df_columns):
    q = str(query_col).strip().lower()
    # exact match
    for col in df_columns:
        if col.lower() == q:
            return col
    # contains
    for col in df_columns:
        if q in col.lower():
            return col
    # words intersection
    q_words = set(re.findall(r'\w+', q))
    best = None
    best_score = 0
    for col in df_columns:
        col_words = set(re.findall(r'\w+', col.lower()))
        score = len(q_words & col_words)
        if score > best_score:
            best_score = score
            best = col
    if best_score > 0:
        return best
    # fuzzy fallback: startswith
    for col in df_columns:
        if col.lower().startswith(q):
            return col
    return None

# Helper: try to convert series to numeric if possible
def ensure_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    else:
        return pd.to_numeric(series, errors='coerce')

# Core: parse and answer query
def answer_query(query: str, df: pd.DataFrame):
    query = query.strip().lower()
    cols = list(df.columns)

    # Patterns
    # how many rows
    if re.search(r'how many rows|number of rows|size of (file|dataset)|how many records', query):
        return f"The dataset contains {len(df):,} rows.", None

    # list columns
    if re.search(r'what columns|list columns|show columns|what are the columns', query):
        return f"The dataset has {len(cols)} columns: {', '.join(cols[:20])}{'...' if len(cols)>20 else ''}.", None

    # describe dataset
    if re.search(r'\bdescribe\b|\bsummary\b|\babstract\b|\boverview\b', query):
        # produce short summarizing paragraph (reuse report generator)
        return generate_paragraph_report(df), None

    # plot x vs y
    m = re.search(r'plot (.+) vs (.+)', query)
    if m:
        raw_x = m.group(2).strip()
        raw_y = m.group(1).strip()
        col_x = find_column_by_name(raw_x, cols)
        col_y = find_column_by_name(raw_y, cols)
        if not col_x or not col_y:
            return f"Couldn't find columns matching '{raw_x}' or '{raw_y}'. Try exact column names.", None
        # produce plot: choose line if x is datetime or sorted numeric, else scatter
        x_series = df[col_x]
        y_series = df[col_y]
        # try to parse x as datetime
        try:
            x_dt = pd.to_datetime(x_series, errors='coerce')
            if x_dt.notnull().sum() > (len(df)*0.2):
                fig = px.line(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}")
                return f"Plotting {col_y} vs {col_x}.", fig
        except Exception:
            pass
        # fallback scatter
        fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}")
        return f"Plotting {col_y} vs {col_x}.", fig

    # average of column
    m = re.search(r'average of (.+)|mean of (.+)', query)
    if m:
        col_raw = (m.group(1) or m.group(2) or '').strip()
        col = find_column_by_name(col_raw, cols)
        if not col:
            return f"Couldn't identify column '{col_raw}'.", None
        series = ensure_numeric(df[col])
        val = series.mean(skipna=True)
        if pd.isna(val):
            return f"Column '{col}' does not appear numeric or has no valid numeric values.", None
        return f"Average of **{col}** = {val:.4f}", None

    # sum
    m = re.search(r'sum of (.+)|total of (.+)', query)
    if m:
        col_raw = (m.group(1) or m.group(2) or '').strip()
        col = find_column_by_name(col_raw, cols)
        if not col:
            return f"Couldn't identify column '{col_raw}'.", None
        series = ensure_numeric(df[col])
        val = series.sum(skipna=True)
        if pd.isna(val):
            return f"Column '{col}' does not appear numeric or has no valid numeric values.", None
        return f"Sum of **{col}** = {val:.4f}", None

    # min / max
    m = re.search(r'(minimum|min|lowest) of (.+)', query)
    if m:
        col_raw = m.group(2).strip()
        col = find_column_by_name(col_raw, cols)
        if not col:
            return f"Couldn't identify column '{col_raw}'.", None
        series = ensure_numeric(df[col])
        val = series.min(skipna=True)
        return f"Minimum of **{col}** = {val}", None
    m = re.search(r'(maximum|max|highest) of (.+)', query)
    if m:
        col_raw = m.group(2).strip()
        col = find_column_by_name(col_raw, cols)
        if not col:
            return f"Couldn't identify column '{col_raw}'.", None
        series = ensure_numeric(df[col])
        val = series.max(skipna=True)
        return f"Maximum of **{col}** = {val}", None

    # how many unique / unique values
    m = re.search(r'(how many unique|unique values|distinct) of (.+)', query)
    if m:
        col_raw = m.group(2).strip()
        col = find_column_by_name(col_raw, cols)
        if not col:
            return f"Couldn't identify column '{col_raw}'.", None
        unique_count = df[col].nunique(dropna=True)
        top_vals = df[col].value_counts().head(5)
        top_str = "; ".join([f"{i}: {v:,}" for i, v in zip(top_vals.index.astype(str), top_vals.values)])
        return f"Column **{col}** has {unique_count:,} unique values. Top values: {top_str}.", None

    # describe column
    m = re.search(r'describe (.+)|summary of (.+)|show stats of (.+)', query)
    if m:
        col_raw = (m.group(1) or m.group(2) or m.group(3) or '').strip()
        col = find_column_by_name(col_raw, cols)
        if not col:
            return f"Couldn't identify column '{col_raw}'.", None
        if pd.api.types.is_numeric_dtype(df[col]) or pd.to_numeric(df[col], errors='coerce').notnull().sum() > 0:
            s = ensure_numeric(df[col]).describe().to_dict()
            stats = ", ".join([f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}" for k, v in s.items()])
            return f"Numeric summary for **{col}** — {stats}", None
        else:
            vc = df[col].value_counts().head(10)
            vc_str = "; ".join([f"{i}: {v}" for i, v in zip(vc.index.astype(str), vc.values)])
            return f"Categorical summary for **{col}** — top values: {vc_str}", None

    # text length stats
    m = re.search(r'(length|text length|characters) of (.+)', query)
    if m:
        col_raw = m.group(2).strip()
        col = find_column_by_name(col_raw, cols)
        if not col:
            return f"Couldn't identify column '{col_raw}'.", None
        lengths = df[col].astype(str).dropna().map(len)
        return f"Text length for **{col}** — avg: {lengths.mean():.2f}, min: {lengths.min()}, max: {lengths.max()}", None

    # top N rows / show head
    if re.search(r'show (first|head|top) ?(\d+)? rows|show first|show head', query):
        m = re.search(r'(\d+)', query)
        n = int(m.group(1)) if m else 5
        snippet = df.head(n).to_dict(orient='records')
        # return as short text
        preview = df.head(n).to_markdown(index=False)
        return f"Showing first {n} rows:\n\n{preview}", None

    # fallback: try simple column lookup + mean if they typed a column name only
    possible_col = find_column_by_name(query, cols)
    if possible_col:
        # show basic describe
        if pd.api.types.is_numeric_dtype(df[possible_col]) or pd.to_numeric(df[possible_col], errors='coerce').notnull().sum() > 0:
            s = ensure_numeric(df[possible_col]).describe().to_dict()
            stats = ", ".join([f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}" for k, v in s.items()])
            return f"Basic numeric summary for **{possible_col}** — {stats}", None
        else:
            vc = df[possible_col].value_counts().head(10)
            vc_str = "; ".join([f"{i}: {v}" for i, v in zip(vc.index.astype(str), vc.values)])
            return f"Top values for **{possible_col}** — {vc_str}", None

    return "Sorry — I couldn't understand the request. Try 'average of <column>', 'sum of <column>', 'plot <y> vs <x>', 'how many rows', 'describe <column>', or 'how many unique of <column>'.", None

# Generate paragraph-style abstract report
def generate_paragraph_report(df: pd.DataFrame, dataset_category=None):
    rows, cols = df.shape
    summary_parts = []
    summary_parts.append(f"The dataset contains {rows:,} records across {cols} columns.")
    
    # Add dataset category information if provided
    if dataset_category:
        summary_parts.append(f"This is a {dataset_category} dataset.")
    
    # detect date column
    date_col = None
    for c in df.columns:
        if 'date' in c.lower():
            try:
                temp = pd.to_datetime(df[c], errors='coerce')
                if temp.notnull().sum() > 0:
                    date_col = c
                    df[c] = temp
                    break
            except Exception:
                continue
    if date_col:
        dmin = df[date_col].min()
        dmax = df[date_col].max()
        if pd.notna(dmin) and pd.notna(dmax):
            summary_parts.append(f"It spans from {dmin.date()} to {dmax.date()}, providing a temporal view of the data useful for trend and time-series analyses.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    if numeric_cols:
        # top 3 numeric by variance
        variances = {c: df[c].var() for c in numeric_cols if df[c].notnull().sum()>0}
        top_vars = sorted(variances.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)[:3]
        top_var_names = [t[0] for t in top_vars]
        sample_nums = ', '.join(numeric_cols[:5])
        summary_parts.append(f"Key numeric attributes include {sample_nums}{' and more' if len(numeric_cols)>5 else ''}. Columns with notable variance include {', '.join(top_var_names)}.")
    if cat_cols:
        sample_cats = ', '.join(cat_cols[:5])
        summary_parts.append(f"Categorical/grouping columns include {sample_cats}{' and others' if len(cat_cols)>5 else ''}, useful for segmentation and grouping analyses.")
    # missing / duplicates
    total_missing = int(df.isnull().sum().sum())
    dup = int(df.duplicated().sum())
    if total_missing > 0:
        summary_parts.append(f"The dataset contains {total_missing:,} missing values across columns; consider cleaning or imputing missing data for robust analysis.")
    if dup > 0:
        summary_parts.append(f"There are {dup:,} duplicate rows which might require deduplication depending on the use-case.")
    # suggested uses
    suggestions = []
    if date_col and len(numeric_cols)>0:
        suggestions.append("time-series trend analysis")
    if len(numeric_cols) >= 2:
        suggestions.append("correlation and regression modeling")
    if len(cat_cols) > 0:
        suggestions.append("segment-level summaries and categorical comparisons")
    if suggestions:
        summary_parts.append("This dataset is well-suited for " + ", ".join(suggestions) + ".")
    else:
        summary_parts.append("This dataset can be explored for descriptive statistics and basic visualizations.")
    return " ".join(summary_parts)

# Alternative text-based report export
def generate_text_report(df: pd.DataFrame, dataset_category=None):
    report = "DATASET ANALYSIS REPORT\n"
    report += "=" * 50 + "\n\n"
    
    report += f"Dataset Overview: {df.shape[0]} rows × {df.shape[1]} columns\n"
    if dataset_category:
        report += f"Dataset Category: {dataset_category}\n"
    report += "\n"
    
    # Add the paragraph report
    report += generate_paragraph_report(df, dataset_category) + "\n\n"
    
    report += "COLUMN HIGHLIGHTS\n"
    report += "-" * 50 + "\n\n"
    
    # Top missing columns
    missing = df.isnull().sum()
    top_missing = missing[missing > 0].sort_values(ascending=False).head(5)
    if not top_missing.empty:
        report += "Columns with most missing values:\n"
        for col, cnt in top_missing.items():
            report += f"- {col}: {cnt} missing ({cnt/len(df)*100:.2f}%)\n"
        report += "\n"
    
    # Numeric extremes
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        report += "Numeric highlights (min / mean / max):\n"
        for c in numeric_cols[:10]:
            s = ensure_numeric(df[c])
            report += f"- {c}: min={s.min():.4f}, mean={s.mean():.4f}, max={s.max():.4f}\n"
        report += "\n"
    
    # Categorical top items
    cat_cols = df.select_dtypes(include=['object','category','string']).columns.tolist()
    if cat_cols:
        report += "Categorical example top values:\n"
        for c in cat_cols[:5]:
            vc = df[c].value_counts().head(5)
            vc_str = ", ".join([f"{i} ({v})" for i, v in zip(vc.index.astype(str), vc.values)])
            report += f"- {c}: {vc_str}\n"
    
    return report

# Sidebar: upload & options
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    # Dataset category selection
    st.markdown("---")
    st.markdown("### Dataset Category")
    dataset_category = st.selectbox(
        "Select dataset category",
        ["General", "Marketing", "Banking", "Healthcare", "Education", "Finance", "Retail", "Other"],
        help="Select the category that best describes your dataset"
    )
    
    st.markdown("---")
    st.markdown("Choose analysis features:")
    show_summary = st.checkbox("Data Summary (Overview)", value=True)
    show_visualization = st.checkbox("Visualization", value=True)
    show_encoding = st.checkbox("Encoding", value=True)
    show_chatbot = st.checkbox("Chatbot (Ask questions)", value=True)
    show_report = st.checkbox("Abstract Report (Paragraph)", value=True)
    
    if st.button("Load Example Dataset"):
        # small example
        np.random.seed(1)
        example_data = {
            'Date': pd.date_range('2023-01-01', periods=120),
            'Open': np.random.normal(100, 10, 120),
            'High': np.random.normal(105, 12, 120),
            'Low': np.random.normal(95, 8, 120),
            'Close': np.random.normal(100, 11, 120),
            'Volume': np.random.randint(1000, 10000, 120),
            'Region': np.random.choice(['North','South','East','West'], 120),
            'Feedback': np.random.choice(['Good','Average','Bad'], 120)
        }
        st.session_state.df = pd.DataFrame(example_data)
        st.session_state.dataset_category = "Finance"  # Set example category
        st.experimental_rerun()

# Load file if uploaded
if uploaded_file:
    try:
        df = load_dataframe(uploaded_file)
        st.session_state.df = df
        st.session_state.dataset_category = dataset_category
        st.success("File loaded.")
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# If we have df
if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df.copy()
    dataset_category = st.session_state.get('dataset_category', 'General')
    
    st.subheader(f"Dataset Preview — {df.shape[0]} rows × {df.shape[1]} columns")
    if dataset_category != 'General':
        st.caption(f"Category: {dataset_category}")
    st.dataframe(df.head(100))

    # Tabs: Overview, Visualize, Encoding, Chatbot, Report
    tabs = st.tabs(["Overview", "Visualize", "Encoding", "Chatbot", "Report"])

    # Overview
    with tabs[0]:
        st.markdown("### Overview & Quick Stats")
        if show_summary:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data types**")
                st.dataframe(pd.DataFrame(df.dtypes, columns=['dtype']).T)
                st.write("**Preview**")
                st.dataframe(df.head(10))
            with col2:
                st.write("**Numeric stats**")
                st.dataframe(df.describe().T)
                missing = pd.DataFrame(df.isnull().sum(), columns=['missing'])
                missing['%'] = (missing['missing'] / len(df)) * 100
                st.write("**Missing values (top 20)**")
                st.dataframe(missing.sort_values('missing', ascending=False).head(20))

    # Visualize
    with tabs[1]:
        st.markdown("### Visualization")
        if show_visualization:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

            col1, col2 = st.columns(2)
            with col1:
                chart = st.selectbox("Chart type", ["Line", "Bar", "Scatter", "Histogram", "Box", "Pie", "Heatmap"])
            with col2:
                if chart in ["Line", "Bar", "Scatter"]:
                    x = st.selectbox("X axis", df.columns, index=0)
                    y = st.selectbox("Y axis (numeric)", numeric) if numeric else st.selectbox("Y axis", df.columns)
                elif chart == "Histogram":
                    x = st.selectbox("Column (numeric)", numeric)
                    y = None
                elif chart == "Box":
                    x = st.selectbox("Category (or column)", categorical if categorical else df.columns)
                    y = st.selectbox("Value (numeric)", numeric)
                elif chart == "Pie":
                    x = st.selectbox("Category", categorical if categorical else df.columns)
                    y = None
                else:
                    x = y = None

            if st.button("Generate Chart"):
                st.markdown("#### Chart Result")
                if chart == "Line":
                    fig = px.line(df, x=x, y=y, title=f"{y} over {x}")
                    st.plotly_chart(fig)
                elif chart == "Bar":
                    fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
                    st.plotly_chart(fig)
                elif chart == "Scatter":
                    color = st.selectbox("Color by (optional)", [None] + df.columns.tolist())
                    fig = px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")
                    st.plotly_chart(fig)
                elif chart == "Histogram":
                    fig = px.histogram(df, x=x, nbins=30, title=f"Histogram of {x}")
                    st.plotly_chart(fig)
                elif chart == "Box":
                    fig = px.box(df, x=x, y=y, title=f"Box plot of {y} by {x}")
                    st.plotly_chart(fig)
                elif chart == "Pie":
                    vc = df[x].value_counts().nlargest(10)
                    fig = px.pie(values=vc.values, names=vc.index, title=f"Pie chart of {x} (top 10)")
                    st.plotly_chart(fig)
                elif chart == "Heatmap":
                    num = df.select_dtypes(include=[np.number])
                    if num.shape[1] < 2:
                        st.warning("Need at least 2 numeric columns for correlation heatmap.")
                    else:
                        fig, ax = plt.subplots(figsize=(10, 7))
                        sns.heatmap(num.corr(), annot=True, ax=ax, cmap='coolwarm', center=0)
                        st.pyplot(fig)

    # Encoding
    with tabs[2]:
        st.markdown("### Encoding Options")
        if show_encoding:
            categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            if categorical_cols:
                st.write("Detected categorical columns:", categorical_cols)
                enc_method = st.radio("Encoding method for categorical features", ["One-Hot Encoding", "Label Encoding"], index=0)
                
                if st.button("Apply Encoding"):
                    if enc_method == "Label Encoding":
                        le = LabelEncoder()
                        for c in categorical_cols:
                            # Handle missing values before encoding
                            df[c] = df[c].fillna('Missing')
                            df[c] = le.fit_transform(df[c].astype(str))
                        st.success("Label Encoding applied successfully!")
                    else:
                        # One-Hot Encoding
                        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=True)
                        st.success("One-Hot Encoding applied successfully!")
                    
                    st.session_state.df = df
                    st.experimental_rerun()
            else:
                st.info("No categorical columns detected.")

    # Chatbot
    with tabs[3]:
        st.markdown("### Chat with Dataset")
        if show_chatbot:
            st.markdown("Ask questions about the loaded dataset in plain English. Examples: 'average of Close', 'plot Close vs Date', 'how many rows', 'describe Volume', 'how many unique of Region'.")
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []  # list of (user, bot) tuples

            # Display chat history
            for user_msg, bot_msg in st.session_state.chat_history:
                st.markdown(f"**You:** {user_msg}")
                if isinstance(bot_msg, str):
                    st.markdown(f"**Bot:** {bot_msg}")
                else:
                    # bot_msg can be tuple (text, fig)
                    text, fig = bot_msg
                    st.markdown(f"**Bot:** {text}")
                    if fig is not None:
                        st.plotly_chart(fig)

            user_input = st.text_input("Enter your question", key="chat_input")
            if st.button("Ask"):
                if not user_input or not user_input.strip():
                    st.warning("Please type a question.")
                else:
                    answer_text, fig = answer_query(user_input, df)
                    # save history
                    if fig is None:
                        st.session_state.chat_history.append((user_input, answer_text))
                        st.markdown(f"**You:** {user_input}")
                        st.markdown(f"**Bot:** {answer_text}")
                    else:
                        st.session_state.chat_history.append((user_input, (answer_text, fig)))
                        st.markdown(f"**You:** {user_input}")
                        st.markdown(f"**Bot:** {answer_text}")
                        st.plotly_chart(fig)

            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.experimental_rerun()

    # Report
    with tabs[4]:
        st.markdown("### Abstract-style Report")
        if show_report:
            paragraph = generate_paragraph_report(df, dataset_category)
            st.markdown(paragraph)

            st.markdown("#### Column Highlights")
            # Show a few highlighted findings in bullet points
            # Top missing columns
            missing = df.isnull().sum()
            top_missing = missing[missing > 0].sort_values(ascending=False).head(5)
            if not top_missing.empty:
                st.markdown("**Columns with most missing values:**")
                for col, cnt in top_missing.items():
                    st.markdown(f"- **{col}**: {cnt} missing ({cnt/len(df)*100:.2f}%)")
            else:
                st.markdown("No missing values detected.")

            # Numeric extremes
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.markdown("**Numeric highlights (min / mean / max)**")
                highlights = []
                for c in numeric_cols[:10]:
                    s = ensure_numeric(df[c])
                    highlights.append(f"- **{c}**: min={s.min():.4f}, mean={s.mean():.4f}, max={s.max():.4f}")
                st.markdown("\n".join(highlights))
            # Categorical top items
            cat_cols = df.select_dtypes(include=['object','category','string']).columns.tolist()
            if cat_cols:
                st.markdown("**Categorical example top values**")
                for c in cat_cols[:5]:
                    vc = df[c].value_counts().head(5)
                    vc_str = ", ".join([f"{i} ({v})" for i, v in zip(vc.index.astype(str), vc.values)])
                    st.markdown(f"- **{c}**: {vc_str}")
            
            # Text export
            st.markdown("---")
            st.markdown("### Export Report")
            text_report = generate_text_report(df, dataset_category)
            st.download_button(
                label="Download Text Report",
                data=text_report,
                file_name="dataset_analysis_report.txt",
                mime="text/plain"
            )

else:
    st.markdown("### No dataset loaded")
    st.markdown("Upload a CSV or click 'Load Example Dataset' from the sidebar to begin.")