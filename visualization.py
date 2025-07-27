import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from scipy.stats import gaussian_kde

# Initialize Gemini API
GENAI_API_KEY = "AIzaSyDWqKiiG3etvFCVIk4_GTuiVTvqK45VUrc"
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
 

import os
import streamlit as st


def call_gemini_api(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error calling Gemini API]: {e}"

#univariate functions
def get_uni_stats(series: pd.Series):
    """Return dict of min, max, mean, median, std for numeric; top counts for categorical."""
    stats = {}
    if np.issubdtype(series.dtype, np.number):
        stats.update({
            "min": round(series.min(),2),
            "max": round(series.max(),2),
            "mean": round(series.mean(),2),
            "median": round(series.median(),2),
            "std": round(series.std(),2),
        })
    else:
        vc = series.value_counts().head(4)
        stats["top"] = vc.to_dict()  # {cat: count, ...}
    return stats

def summarize_uni(stats: dict, is_num: bool):
    """Return a short human summary segment for the AI prompt."""
    if is_num:
        return (
            f"Range: {stats['min']}–{stats['max']}; "
            f"Mean: {stats['mean']}; Median: {stats['median']}; Std: {stats['std']}"
        )
    else:
        pairs = "; ".join(f"{k}({v})" for k,v in stats["top"].items())
        return f"Top categories: {pairs}"
    
#bivariate functions

def get_biv_stats(df, x_col, y_col):
    x, y = df[x_col].dropna(), df[y_col].dropna()
    stats = {"x_min": None, "x_max": None, "y_min": None, "y_max": None, "corr": "N/A"}
    corr = None

    # numeric vs numeric
    if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
        corr = round(x.corr(y), 3)
        stats.update({
            "x_min": round(x.min(),2), "x_max": round(x.max(),2),
            "y_min": round(y.min(),2), "y_max": round(y.max(),2),
            "corr": corr
        })
    return stats, corr

def summarize_corr(corr):
    if corr is None:
        return "N/A"
    if abs(corr) >= 0.7: strength = "strong"
    elif abs(corr) >= 0.4: strength = "moderate"
    else: strength = "weak"
    direction = "positive" if corr>0 else "negative" if corr<0 else "no"
    return f"{strength} {direction} (r={corr})"

def summarize_density(x, y):
    try:
        kde_vals = gaussian_kde(np.vstack([x,y]))(np.vstack([x,y]))
        return "Dense central cluster" if np.percentile(kde_vals,90)>2*np.median(kde_vals) else "Even spread"
    except:
        return "N/A"

# ——— Helper for Multivariate Stats & Summary ———
def get_multi_stats(df: pd.DataFrame, cols: list[str]):
    """Returns min/max/mean/median for each numeric col, and the top 3 corr pairs."""
    stats = {}
    # basic per-column summary
    for c in cols:
        s = df[c].dropna()
        stats[c] = {
            "min": round(s.min(),2),
            "max": round(s.max(),2),
            "mean": round(s.mean(),2),
            "median": round(s.median(),2)
        }
    # top 3 correlations
    corr_mat = df[cols].corr().abs().unstack()
    # remove self‐correlations
    corr_mat = corr_mat[corr_mat < 1.0]
    top_pairs = (corr_mat
        .sort_values(ascending=False)
        .drop_duplicates()
        .head(3)
        .index
        .tolist()
    )
    top_corrs = [f"{a}-{b} (r={round(df[a].corr(df[b]),2)})" for a,b in top_pairs]
    return stats, top_corrs

def summarize_multi(stats: dict, top_corrs: list[str]):
    """Return a one‐liner summary block for AI prompt."""
    col_summaries = "; ".join(
        f"{c}: [{v['min']}–{v['max']}, μ={v['mean']}, med={v['median']}]" 
        for c,v in stats.items()
    )
    corr_summary = "; ".join(top_corrs)
    return col_summaries, corr_summary


def save_and_report(fig, title, insight_text, filename=None):
    os.makedirs("plots", exist_ok=True)
    if not filename:
        sanitized = title.replace(" ", "_").replace(":", "").replace(",", "")
        filename = f"plots/{sanitized}.png"
    fig.savefig(filename)
    sec = {"title": title, "image": filename, "text": insight_text}
    st.session_state.setdefault('report_sections', []).append(sec)
    st.success("✅ Added to report!")
    

def show_report_preview():
    st.header("📝 AI Report Preview")

    report = st.session_state.get("report_sections", [])

    if not report:
        st.info("No report sections added yet.")
        return

    for i, section in enumerate(report):
        st.markdown(f"### {i+1}. {section['title']}")
        if "image" in section:
            st.image(section["image"], caption=section.get("title", "AI Report"), use_container_width=True)
        else:
            st.warning(f"⚠️ No image for section: {section.get('title', 'Untitled Section')}")
        st.markdown("**AI Insight:**")
        if "text" in section:
            st.info(section["text"])
        else:
            st.warning(f"⚠️ No insight for section: {section.get('title', 'Unknown')}")
        st.markdown("---")

#helper function for Advance plot:

def top_correlations(df: pd.DataFrame, cols: list[str], n: int = 5):
    """Return top n unique corr pairs as strings 'A-B (r=0.87)'."""
    corr = df[cols].corr().abs().unstack()
    corr = corr[corr < 1.0].drop_duplicates().sort_values(ascending=False)
    top = corr.head(n).index.tolist()
    return [f"{a}-{b} (r={round(df[a].corr(df[b]),2)})" for a,b in top]




def show():
    st.title("📊 Data Visualization")

    # Check if cleaned dataset is available
    if 'dataframe' not in st.session_state:
        st.warning("⚠️ No cleaned dataset found! Please complete the Data Cleaning process first.")
        return

    # Load cleaned dataset
    df = st.session_state['dataframe']

    # Sidebar for selecting visualization type
    st.sidebar.header("📌 Select Visualization Type")
    vis_type = st.sidebar.radio(
        "Choose a type:", 
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Advanced Visualizations"]
    )

#_____________________________________________________________________________________________________________________________________________
 
    # Univariate Analysis (Single Column)
    if vis_type == "Univariate Analysis":
        st.subheader("📈 Univariate Analysis")

        # 1) Column selector
        cols = df.columns.tolist()
        col = st.selectbox("Select Column", cols, key="uni_col")
        series = df[col].dropna()
        is_num = np.issubdtype(series.dtype, np.number)

        # 2) Plot
        fig, ax = plt.subplots()
        if is_num:
            sns.histplot(series, kde=True, bins=30, ax=ax)
            graph_type = "Histogram + KDE"
        else:
            sns.countplot(data=df, x=col, order=series.value_counts().index, ax=ax)
            graph_type = "Count Plot"
            plt.xticks(rotation=45)
        ax.set_title(f"{graph_type}: {col}")
        st.pyplot(fig)

        # 3) Generate AI Insight button
        insight = None
        if st.button("🧠 Generate AI Insight for Univariate", key=f"gen_uni_{col}"):
            stats   = get_uni_stats(series)
            summary = summarize_uni(stats, is_num)
            if is_num:
                prompt = f"""
                You are a data assistant. Given a {graph_type} of '{col}':
                • {summary}

                Write **3–4 very short** bullet insights about skewness, peaks, spread, outliers.
                Avoid recommendations.
                """.strip()
            else:
                prompt = f"""
                You are a data assistant. Given a {graph_type} of '{col}':
                • {summary}

                Write **3–4 very short** bullet insights about category distribution and rare values.
                Avoid recommendations.
                """.strip()

            try:
                raw = call_gemini_api(prompt)
                # keep only up to 4 bullet lines
                lines = [l for l in raw.split("\n") if l.strip().startswith("-")][:4]
                insight = "\n".join(lines) if lines else raw

                st.session_state[f"insight_{col}"] = insight
                st.session_state[f"fig_{col}"] = fig
            except Exception as e:
                st.error(f"❌ AI call failed: {e}")

        # 4) Display AI Insight (if generated)
        insight_key = f"insight_{col}"
        fig_key = f"fig_{col}"
        if insight_key in st.session_state:
            st.markdown("**📌 AI Insight:**")
            st.info(st.session_state[insight_key])

            if st.button("📄 Add to Report", key=f"add_uni_{col}"):
                st.write("YES YES")  # ✅ this will now execute
                save_and_report(
                    st.session_state[fig_key], f"Univariate: {col}", st.session_state[insight_key]
                )
                st.success("✅ Univariate section added to report!")
#_____________________________________________________________________________________________________________________________________________
    # Bivariate Analysis (Two Columns)
    elif vis_type == "Bivariate Analysis":
        st.subheader("🔀 Bivariate Analysis")

        cols = df.columns.tolist()
        x_col = st.selectbox("X‑axis", cols, key="x_biv")
        y_col = st.selectbox("Y‑axis", cols, key="y_biv")
        biv_key = f"{x_col}_vs_{y_col}" 
        # Choose plot type based on dtypes
        x_num = np.issubdtype(df[x_col].dtype, np.number)
        y_num = np.issubdtype(df[y_col].dtype, np.number)

        if x_num and y_num:
            opts = ["Scatter Plot","Line Plot","Regression Plot"]
        elif not x_num and y_num:
            opts = ["Box Plot","Bar Plot"]
        elif x_num and not y_num:
            opts = ["Box Plot (hue)","Bar Plot (hue)"]
        else:
            opts = ["Count Plot","Crosstab Heatmap"]

        graph_type = st.selectbox("Plot Type", opts, key="biv_graph")
        fig, ax = plt.subplots()

        # Rendering
        if graph_type=="Scatter Plot":
            sns.scatterplot(data=df,x=x_col,y=y_col,ax=ax)
        elif graph_type=="Line Plot":
            sns.lineplot(data=df,x=x_col,y=y_col,ax=ax)
        elif graph_type=="Regression Plot":
            sns.regplot(data=df,x=x_col,y=y_col,ax=ax)
        elif graph_type=="Box Plot":
            sns.boxplot(data=df,x=x_col,y=y_col,ax=ax)
        elif graph_type=="Bar Plot":
            sns.barplot(data=df,x=x_col,y=y_col,ax=ax)
        elif graph_type=="Box Plot (hue)":
            sns.boxplot(data=df,x=y_col,y=x_col,ax=ax)
        elif graph_type=="Bar Plot (hue)":
            sns.barplot(data=df,x=y_col,y=x_col,ax=ax)
        elif graph_type=="Count Plot":
            sns.countplot(data=df,x=x_col,hue=y_col,ax=ax)
        else:  # Crosstab Heatmap
            ct = pd.crosstab(df[x_col],df[y_col])
            sns.heatmap(ct,annot=True,fmt="d",cmap="Blues",ax=ax)

        ax.set_title(f"{graph_type}: {x_col} vs {y_col}")
        st.pyplot(fig)

        # AI Insight
        if st.button("🧠 Generate AI Insight"):
            stats, corr = get_biv_stats(df,x_col,y_col)
            corr_sum = summarize_corr(corr)
            density = summarize_density(df[x_col].dropna(), df[y_col].dropna())

            # Build prompt differently for cat‑cat
            if not x_num and not y_num:
                top_pairs = pd.crosstab(df[x_col],df[y_col]).stack() \
                            .sort_values(ascending=False).head(3).index.tolist()
                pairs_str = "; ".join(f"{a}-{b}" for a,b in top_pairs)
                prompt = f"""
                    You are an AI assistant. Given a categorical bivariate plot between '{x_col}' and '{y_col}', provide 3‑5 bullet insights about:
                    - Most common category pairs: {pairs_str}
                    - Any rare combinations or imbalances
                    - Overall distribution patterns

                    
                    Write only 3-4 **short** bullet insights.
                    Avoid recommendations or extra words.
                    """
                
            else:
                prompt = f"""
                    You are an AI assistant. Given this '{graph_type}' between '{x_col}' and '{y_col}':
                    - Range X: {stats['x_min']}-{stats['x_max']}
                    - Range Y: {stats['y_min']}-{stats['y_max']}
                    - Correlation: {corr_sum}
                    - Density: {density}

                    Provide 3-5 concise bullet insights about trends, correlation strength, and cluster patterns.
                    
                    Write only 3-4 **short** bullet insights.
                    Avoid recommendations or extra words.
                    """

            try:
                raw = call_gemini_api(prompt)
                # keep only up to 4 bullet lines
                lines = [l for l in raw.split("\n") if l.strip().startswith("-")][:4]
                insight = "\n".join(lines) if lines else raw

                st.session_state[f"insight_{biv_key}"] = insight
                st.session_state[f"fig_{biv_key}"] = fig
            except Exception as e:
                st.error(f"❌ AI call failed: {e}")

        # 4) Display AI Insight (if generated)
        insight_key = f"insight_{biv_key}"
        fig_key = f"fig_{biv_key}"

        if insight_key in st.session_state:
            st.markdown("**📌 AI Insight:**")
            st.info(st.session_state[insight_key])

            if st.button("📄 Add to Report", key=f"add_biv_{biv_key}"):
                save_and_report(
                    st.session_state[fig_key],
                    f"Bivariate: {x_col} vs {y_col}",
                    st.session_state[insight_key]
                )
                st.success("✅ Bivariate section added to report!")
#__________________________________________________________________________________________________________________________________
    elif vis_type == "Multivariate Analysis":
        st.subheader("📊 Multivariate Analysis")

        # 1. Column selection
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        selected = st.multiselect("Select ≥2 numeric columns", numeric_cols, key="multi_cols")
        
        key = None
        insight = None
        fig = None
        prompt = None

        if selected:
            key = "multivariate_" + "_".join(selected)


        if len(selected) < 2:
            st.warning("Please select at least two numeric columns.")
            st.stop()

        key = "multivariate_" + "_".join(selected)

        # 2. Plot
        fig = sns.pairplot(df[selected])
        st.pyplot(fig)

        # 3. AI‑Insight button
        if st.button("🧠 Generate AI Insight for Multivariate"):
            # compute stats
            stats, top_corrs = get_multi_stats(df, selected)
            col_summ, corr_summ = summarize_multi(stats, top_corrs)

            # build concise prompt
            prompt = f"""
                You are a data assistant. Given a Pairplot of columns: {', '.join(selected)}.

                • Column summaries:
                {col_summ}

                • Top correlations:
                {corr_summ}

                Write **3–4 very short** bullet insights about patterns, clusters, or strong relationships. No recommendations.
                """.strip()

            try:
                raw = call_gemini_api(prompt)
                bullets = [l for l in raw.split("\n") if l.strip().startswith("-")][:4]
                insight = "\n".join(bullets) if bullets else raw

                st.markdown("**📌 AI Insight:**")
                st.info(insight)

                # store in session state
                st.session_state[f"{key}_insight"] = insight
                st.session_state[f"{key}_fig"] = fig

            except Exception as e:
                st.error(f"❌ AI call failed: {e}")

        # 5. Add to Report if insight exists
        if f"{key}_insight" in st.session_state:
            if st.button("📄 Add to Report", key=f"add_multi_{key}"):
                save_and_report(
                    st.session_state[f"{key}_fig"],
                    f"Multivariate: {', '.join(selected)}",
                    st.session_state[f"{key}_insight"]
                )
                st.success("✅ Multivariate plot added to report!")
                
    # Advanced Visualizations (Pairplot, Heatmap)
        
    elif vis_type == "Advanced Visualizations":
        st.subheader("📌 Advanced Visualizations")

        plot_type = st.selectbox("Choose Plot Type",
            ["Pairplot", "Correlation Heatmap", "Custom Heatmap"], key="adv_plot")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        fig = None

        # — Pairplot —
        if plot_type == "Pairplot":
            sample = df[numeric_cols]
            if len(sample) > 500:
                sample = sample.sample(500, random_state=1)
            fig = sns.pairplot(sample)
            key = "advanced_pairplot"
            prompt = f"""
                You are a data assistant. This Pairplot shows relationships between columns: {', '.join(numeric_cols)}.
                • Diagonal = distributions; off‑diagonal = scatter relationships.
                • No trendlines or aggregations.
                Write 3–4 very brief bullet insights about clusters, correlations, or outliers.
                dont include recommendations.
                """.strip()

        # — Correlation Heatmap —
        elif plot_type == "Correlation Heatmap":
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            key = "advanced_corr_heatmap"
            top = top_correlations(df, numeric_cols, n=5)
            prompt = f"""
                You are a data assistant. This heatmap shows Pearson correlations among {', '.join(numeric_cols)}.
                • Top correlations: {"; ".join(top)}
                Write 3–4 very brief bullet insights about strong/weak pairs or multicollinearity.
                dont include recommendations.
                """.strip()

        # — Custom Heatmap —
        else:  # Custom Heatmap
            cols = st.multiselect("Select columns for Custom Heatmap", numeric_cols, key="custom_cols")
            if len(cols) < 2:
                st.warning("Select at least two columns.")
                st.stop()
            corr = df[cols].corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            key = "advanced_custom_heatmap_" + "_".join(cols)
            top = top_correlations(df, cols, n=5)
            prompt = f"""
                You are a data assistant. This custom heatmap shows Pearson correlations among {', '.join(cols)}.
                • Top correlations: {"; ".join(top)}
                Write 3–4 very brief bullet insights about relationships or surprising pairs.
                dont include recommendations.
                """.strip()

        # Display the figure
        if plot_type == "Pairplot":
            st.pyplot(fig)
        else:
            st.pyplot(fig)

        # — AI Insight & Report —
        if st.button("🧠 Generate AI Insight"):
            if prompt is not None:
                try:
                    raw = call_gemini_api(prompt)
                    bullets = [l for l in raw.split("\n") if l.strip().startswith("-")][:4]
                    insight = "\n".join(bullets) if bullets else raw

                    st.markdown("**📌 AI Insight:**")
                    st.info(insight)

                    # Add to session state for consistent state-saving
                    st.session_state[f"adv_insight_{key}"] = insight
                    st.session_state[f"adv_fig_{key}"] = fig

                except Exception as e:
                    st.error(f"❌ AI call failed: {e}")

        # Show "Add to Report" only if insight exists
        insight_key = f"adv_insight_{key}"
        fig_key = f"adv_fig_{key}"

        if insight_key in st.session_state:
            if st.button("📄 Add to Report", key=f"add_adv_{key}"):
                save_and_report(
                    st.session_state[fig_key],
                    f"Advanced: {plot_type}",
                    st.session_state[insight_key]
                )
                st.success("✅ Advanced visualization added to report!")
    st.success("✅ Visualization Complete! Modify selections to explore more insights.")
    show_report_preview()   