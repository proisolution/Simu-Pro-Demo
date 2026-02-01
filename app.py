import streamlit as st
import pandas as pd
import random
from openai import OpenAI
import os
import io
import time
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Public Health Agent Simulation", layout="wide")

# Try importing geopandas (It might not be installed in all environments)
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Initialize the "Bank" (Session State)
if 'decision_bank' not in st.session_state:
    st.session_state['decision_bank'] = pd.DataFrame()

# Initialize Page State (Default to Playground)
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'playground'

# --- 2. DATA LISTS & MAPPINGS (0, 1, 2 Logic) ---

ages = ['under 50', 'over 50']
genders = ['Female', 'Male']
races = ["White", "Black", "Asian"]
h_incomes = ['below $70,000', 'above $70,000']
edu_levels = ['High school or less', 'Some college', 'Bachelor or more']
family_d = ['did not report', 'report']
cities = ["Atlanta", "San Francisco"]

# City Context Dictionary (Your Research Data)
city_contexts = {
    "Atlanta": [
        "Atlanta is a major city in the southeastern United States with a diverse population...",
        "Atlanta, Georgia, is a culturally diverse city with notable health disparities...",
        "In Atlanta, the population is highly diverse, including large Black and Hispanic communities..."
    ],
    "San Francisco": [
        "San Francisco is a coastal city in California known for its progressive policies...",
        "San Francisco is known for its tech industry and high-income inequality...",
        "San Francisco’s population is generally responsive to public health guidelines..."
    ]
}

def get_vectors(age, gender, race, edu, inc, fam):
    """
    Maps inputs to Integer IDs (0, 1, 2)
    """
    # Mappings
    map_age = {'under 50': '0', 'over 50': '1'}
    map_gen = {'Female': '0', 'Male': '1'}
    map_inc = {'below $70,000': '0', 'above $70,000': '1'}
    map_fam = {'did not report': '0', 'report': '1'}
    map_race = {'White': '0', 'Black': '1', 'Asian': '2'}
    map_edu = {'High school or less': '0', 'Some college': '1', 'Bachelor or more': '2'}

    # Get digits
    d_age = map_age.get(age, '0')
    d_gen = map_gen.get(gender, '0')
    d_race = map_race.get(race, '0')
    d_edu = map_edu.get(edu, '0')
    d_inc = map_inc.get(inc, '0')
    d_fam = map_fam.get(fam, '0')
    
    # Create Vectors
    vec_demo = f"{d_age}{d_gen}{d_race}{d_edu}{d_inc}"       # 5-digit (Base)
    vec_full = f"{d_age}{d_gen}{d_race}{d_edu}{d_inc}{d_fam}" # 6-digit (Context)
    
    return vec_demo, vec_full

# --- 3. PROMPT GENERATION (Original Text) ---

base_prompt_template = """Imagine yourself in the following situation: [{SITU}].

# City Context
{CITY_CONTEXT}

# Personal Risk: Information provided by public health authorities at this time suggest that the mortality rate is around 1%. Almost all people who die of the disease are over the age of 75 years. Young people almost never die from the disease, but they appear to be able to transmit the disease to others. For younger people, symptomatic disease results in an influenza-like illness lasting around 1 week, with a quarantine period of around 2 weeks where an infected person is not allowed to go outside their house. It also appears that at least 50% of people who are infected do not get any symptoms and do not know they are infected.

{NEWS_BLOCK}
Your background and personal circumstances are as follows: 
[You are {AGE} years old, {GENDER} of {RACE} ethnicity{HISPANIC}, living in {CITY}. 
Your income is {H_INCOME}. Your education level is {EDU_LEVEL}.] 
{FAMILY_BLOCK}

Please use this persona to answer the question below:

'How likely are you to get tested and self-report your symptoms to the public health authority if you experience signs of a flu-like symptom, which may make you need to self-quarantine?'

In this context, please think step by step before answering Yes or No based on your persona and the influence of your family member's decision. 
"""

def generate_prompt_text2(age, gender, race, income, edu, fam_decision, city, use_fam, ctx_idx):
    # 1. Contexts
    c_context = city_contexts[city][ctx_idx]
    situation = f"There are currently 10–100 cases in your neighborhood in {city}"
    
    # 2. Dynamic Blocks
    fam_block = f"In your household, another family member recently experienced similar symptoms and decided to **{fam_decision}** their illness to public health authorities." if use_fam else ""
    news_block = "" # (Hidden for simplicity in this specific view, adds clutter)

    # 3. Fill Template
    research_prompt = base_prompt_template.format(
        SITU=situation, CITY_CONTEXT=c_context, NEWS_BLOCK=news_block, FAMILY_BLOCK=fam_block,
        AGE=age, GENDER=gender, RACE=race, HISPANIC="", CITY=city, H_INCOME=income, EDU_LEVEL=edu
    )
    
    # 4. Append Technical Instruction for the machine
    instruction = """
    
    ### INSTRUCTION
    Constraint Checklist:
    1. Decision must be exactly "Yes" or "No".
    2. Reason should be a short sentence summarizing your rationale (no commas inside the reason).
    3. Reporting_Rate must be a number between 0-100.
    4. Confidence must be one of: Very Certain, Somewhat Certain, Uncertain.
    
    Example Output:
    Yes,My family reported so I should too......,85,Somewhat Certain
    """
    return research_prompt + instruction

def mock_batch_generate(age, gender, race, edu, inc, fam, city, count=10):
    """
    Generates 10 Mock agents instantly for the workshop demo.
    """
    results = []
    vec_5, vec_6 = get_vectors(age, gender, race, edu, inc, fam)
    
    # Fake Score Logic (Just to make the demo look realistic)
    base_score = 30
    if fam == 'report': base_score += 30  # Family influence
    if gender == 'Female': base_score += 5
    if age == 'over 50': base_score += 5   
    if race == 'Asian': base_score += 5
    if edu == 'Bachelor or more': base_score += 5
    if inc == 'above $70,000': base_score += 5
    if race == 'Black': base_score -= 5
    if city == 'San Francisco': base_score += 10
    
    for _ in range(count):
        # Random variance
        final_score = min(max(base_score + random.randint(-20, 20), 0), 100)
        decision = "Yes" if final_score > 50 else "No"
        
        results.append({
            "Age": age, "Gender": gender, "Race": race, 
            "Education": edu, "Income": inc, "Family_Influence": fam,
            "Vector_5bit": vec_5, "Vector_6bit": vec_6,
            "Decision": decision
        })
    return pd.DataFrame(results)


# Data Lists
ages = ['under 50', 'over 50']
genders = ['Female', 'Male']
races = ["White", "Black", "Asian"]
incomes = ['below $100,000', 'above $100,000']
h_incomes = ['below $70,000', 'above $70,000']
num_cases = ['0 case', '<10 cases', "10–100 cases", ">100 cases"]
edu_levels = ['High school or less', 'Some college', 'Bachelor or more']
family_d = ['report', 'did not report']
cities = ["Atlanta", "San Francisco"]
hispanic_opts = ['', ' and Hispanic background']

# Your specific City Contexts
city_contexts = {
    "Atlanta": [  # Atlanta
        "Atlanta is a major city in the southeastern United States with a diverse population and a mix of urban and suburban areas. Access to healthcare varies across neighborhoods.",
        "Atlanta, Georgia, is a culturally diverse city with notable health disparities between communities. Some neighborhoods have strong trust in public health systems, while others face barriers due to past systemic inequalities and economic conditions.",
        "In Atlanta, the population is highly diverse, including large Black and Hispanic communities. While some areas benefit from strong local healthcare infrastructure, others face limited access. Past experiences with unequal healthcare delivery have shaped public attitudes, with mistrust of government communication remaining a barrier in certain areas. Public health messaging during pandemics is often met with mixed responses."
    ],
    "San Francisco":  [  # San Francisco
        "San Francisco is a coastal city in California known for its progressive policies and high standard of living. It has strong healthcare infrastructure and high vaccination rates.",
        "San Francisco is known for its tech industry and high-income inequality. The city has a relatively health-aware population, but homeless and underserved communities face significant challenges in accessing care and support during public health crises.",
        "San Francisco’s population is generally responsive to public health guidelines. However, sharp contrasts in income and housing stability lead to unequal healthcare access. While many residents trust government guidance and technology-driven health tracking, marginalized groups often experience exclusion. Public health officials emphasize community outreach to bridge gaps, particularly in Asian and Latino communities."
    ]
}

news_options = [
'Not reporting your symptoms could result in worsening health, delayed treatment, and potential long-term complications.',
        'By reporting your symptoms, you help protect your family, co-workers, and community from further spread of illness.',
        'According to recent data, early reporting of symptoms reduces disease transmission by 40 % and increases the chance of recovery with mild symptoms.',
    'Custom (Write your own)'
]

# --- 2. LOGIC: PROMPT & API ---

# The Base Template (Your Exact Wording)
base_prompt_template = """Imagine yourself in the following situation: [{SITU}].

# City Context
{CITY_CONTEXT}

# Personal Risk: Information provided by public health authorities at this time suggest that the mortality rate is around 1%. Almost all people who die of the disease are over the age of 75 years. Young people almost never die from the disease, but they appear to be able to transmit the disease to others. For younger people, symptomatic disease results in an influenza-like illness lasting around 1 week, with a quarantine period of around 2 weeks where an infected person is not allowed to go outside their house. It also appears that at least 50% of people who are infected do not get any symptoms and do not know they are infected.

{NEWS_BLOCK}
Your background and personal circumstances are as follows: 
[You are {AGE} years old, {GENDER} of {RACE} ethnicity{HISPANIC}, living in {CITY}. 
Your income is {H_INCOME}. Your education level is {EDU_LEVEL}.] 
{FAMILY_BLOCK}

Please use this persona to answer the question below:

'How likely are you to get tested and self-report your symptoms to the public health authority if you experience signs of a flu-like symptom, which may make you need to self-quarantine?'

In this context, please think step by step before answering Yes or No based on your persona and the influence of your family member's decision. 
"""

def generate_prompt_text(age, gender, race, income, edu, cases, fam_decision, city, hispanic, news_txt, use_fam, use_news, ctx_idx):
    # 1. Select City Context String
    c_context = city_contexts[city][ctx_idx]
    
    # 2. Build Dynamic Blocks
    news_block = f"# Recent News\nYou have recently seen the following news: \"{news_txt}\"\n" if use_news and news_txt else ""
    fam_block = f"In your household, another family member recently experienced similar symptoms and decided to **{fam_decision}** their illness to public health authorities." if use_fam else ""
    situation = f"There are currently {cases} in your neighborhood in {city}"
    
    # 3. Fill Research Template
    research_prompt = base_prompt_template.format(
        SITU=situation, CITY_CONTEXT=c_context, NEWS_BLOCK=news_block, FAMILY_BLOCK=fam_block,
        AGE=age, GENDER=gender, RACE=race, HISPANIC=hispanic, CITY=city, H_INCOME=income, EDU_LEVEL=edu
    )
    
    # 4. Append CSV Instruction (Invisible to user conceptual logic, needed for machine)
    technical_instruction = """
    ### SYSTEM INSTRUCTION

    Analyze the persona and situation step-by-step internally.
    Then, output your final decision ONLY as a single CSV row.
    Do not add any other text, labels, or explanations outside the CSV.
    
    The CSV columns are: Decision,Reason,Reporting_Rate,Confidence
    
    Constraint Checklist:
    1. Decision must be exactly "Yes" or "No".
    2. Reason should be a short sentence summarizing your rationale (no commas inside the reason).
    3. Reporting_Rate must be a number between 0-100.
    4. Confidence must be one of: Very Certain, Somewhat Certain, Uncertain.
    
    Example Output:
    Yes,My family reported so I should too......,85,Somewhat Certain

 
    """
    #    In this context, please think step by step before answering Yes or No based on your persona and the influence of your family member's decision. 
    # **Answer:** Yes or No  
    # **SHORT REASON**:.....  
    # Explain to me the rationale behind why you made this decision.  
    # Also, **Reporting rate:** [0–100% based on the persona].  
    # And **Confidence Level:** (Very Certain, Somewhat Certain, Uncertain)

    # output your final decision in a SINGLE CSV LINE.
    # Header: Decision,Reason,Reporting_Rate,Confidence
    return research_prompt + technical_instruction

def validate_and_parse_csv(llm_response):
    try:
        clean_text = llm_response.replace("```csv", "").replace("```", "").strip()
        # Grab last line if multiple lines exist (or parse specifically)
        # For robustness, we look for the line starting with Yes or No or splitting by comma
        header = ["Decision", "Reason", "Reporting_Rate", "Confidence"]
        
        # Simple parser: look for the header, if not found, assume the whole text is the CSV row if it matches structure
        if "Decision,Reason" in clean_text:
            df = pd.read_csv(io.StringIO(clean_text))
        else:
            # Fallback: create DF from raw string if header is missing but data is there
            df = pd.read_csv(io.StringIO(clean_text), names=header)
            
        return True, df
    except:
        return False, None

def run_simulation(api_key, prompt, input_data):
    if not api_key:
        st.warning("⚠️ API Key missing. Using Mock Mode.")
        time.sleep(1)
        # Mock Logic for Demo
        return pd.Series({"Decision": "Yes", "Reason": "Mock Demo Reason", "Reporting_Rate": 75, "Confidence": "High"})

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        
        valid, df = validate_and_parse_csv(content)
        if valid and not df.empty:
            # Merge Inputs + Outputs
            full_row = pd.concat([pd.DataFrame([input_data]), df.iloc[[0]].reset_index(drop=True)], axis=1)
            st.session_state['decision_bank'] = pd.concat([st.session_state['decision_bank'], full_row], ignore_index=True)
            return df.iloc[0]
        else:
            st.error("Invalid CSV from LLM")
            return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

@st.cache_data
def process_raw_data_files_p(mapfolder, folder):
    """
    Adapted from your script. Uses caching to avoid re-processing on every click.
    """
    try:
        REPORT_PATH = folder+'/DiseaseReports.tsv'
        LOG_PATH = folder+'/patterns_of_life.log'

        CHAR_PATH = folder+'/AgentCharacteristicsTable.tsv'
        # process raw data
        raw_disease_reports = pd.read_csv(REPORT_PATH, sep='\t|,', engine='python')
        region = [int(r[1:]) for r in raw_disease_reports['[regionId']]
        raw_disease_reports['regionId'] = region
        regions = raw_disease_reports['regionId'].unique()
        
        # find the considered single bias & population in each region
        biasTypes = ''
        population = []
        for line in open(LOG_PATH).readlines():
            if 'Considered single bias types' in line:
                biasTypes = line.split(':')[-1].strip().split(', ')
            if 'Number of agents in Region' in line:
                population.append(int(line.split(': ')[-1]))
        print(population)
        singleBias = {}
        for bias in biasTypes:
            singleBias[bias] = []
        
        # convert Str type of reports into List of booleans
        #  to represent the single bias report
        for report in raw_disease_reports['Report(Single)]']:
            for bias in biasTypes:
                singleBias[bias].append(bias in report)
        for bias in biasTypes:
            raw_disease_reports[bias] = singleBias[bias]
        
        # Count total number of infected agents in each region
        #  Also count the reported cases in each senario
        data_for_plot = raw_disease_reports[raw_disease_reports['diseaseStatus'] == 'Infectious']    
        data_for_plot = data_for_plot[['step','agentId','regionId','Report(Component)']+biasTypes]

        count_reports = []
        labels = ['regionId','# Real','# Component'] + ['# '+bias for bias in biasTypes]

        for rid in regions:
            rline = [rid]
            rdata = data_for_plot[data_for_plot['regionId'] == rid]
            rline.append(len(rdata))
            rline.append(len(rdata[rdata['Report(Component)'] == True]))
            for bias in biasTypes:
                rline.append(len(rdata[rdata[bias] == True]))
            count_reports.append(rline)
        report_summary = pd.DataFrame.from_records(count_reports, columns = labels)
        
        # Calculate the reporting rate
        report_summary['population'] = population
        for i in range(2, len(labels)):
            report_summary[labels[i].replace('#','%')] = report_summary[labels[i]]/report_summary['population']
        report_summary = report_summary.fillna(0)
        
        # 6. Geo/Census Merge (Only if GeoPandas is available)
        geo_df = pd.DataFrame()
        if HAS_GEOPANDAS:
            shapefile = f"{root_path}/{mapfolder}/region_census.shp"
            try:
                geo_df = gpd.read_file(shapefile)
                # Census Calculations
                geo_df['Age >= 50'] = geo_df['AgeGroup7'] + geo_df['AgeGroup8'] + geo_df['AgeGroup9']
                geo_df['Income >= 50k'] = geo_df['IndiInc3'] + geo_df['IndiInc4'] + geo_df['IndiInc5']
                geo_df['Education > High School'] = geo_df['EduLevel3'] + geo_df['EduLevel4']
                geo_df['Race White'] = geo_df['Race0']
                geo_df['Race Not White'] = 1 - geo_df['Race0']
                geo_df['Female'] = 1 - geo_df['Male']
                
                # Merge Data
                percent_report_labels = [labels[i].replace('#','%') for i in range(2, len(labels))]
                # Assuming index matches or 'id' column matches regionId
                # Simplifying for demo:
                for l in percent_report_labels:
                    if l in report_summary.columns:
                        geo_df[l] = report_summary[l]
                
                geo_df["% Component"] = report_summary["# Component"] / report_summary["# Real"]
            except Exception as e:
                st.warning(f"Shapefile not found at {shapefile}. Showing stats without Map.")
                
        return report_summary, geo_df

    except Exception as e:
        st.error(f"Error processing files: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 3. UI LAYOUT ---

# st.title("🧬 Public Health Agent Simulation")
with st.sidebar:
    st.header("Navigation")
    if st.button("🕹️ Single Agent Playground", use_container_width=True):
        st.session_state['current_page'] = 'playground'
    
    if st.button("📊 Batch Simulation (Data Bank)", use_container_width=True):
        st.session_state['current_page'] = 'batch'
    if st.button("🗺️ Research Outcomes", use_container_width=True):
        st.session_state['current_page'] = 'outcomes'
    
    st.markdown("---")
    # Path Configuration for Tab 3
    st.caption("Admin Settings (Tab 3)")
    root_path = st.text_input("Simulation Root Path", value="./")
    
    st.markdown("---")
    st.info(f"Current Mode: **{st.session_state['current_page'].title()}**")

# --- 4. PAGE LOGIC ---

# ==========================================
# PAGE 1: SINGLE AGENT PLAYGROUND
# ==========================================
if st.session_state['current_page'] == 'playground':
    st.title("🕹️ Single Agent Prompt Explorer")
    st.caption("Step 1: Understand how the Prompt changes based on Inputs.")
# st.caption("Workshop Demo: Generative Agents & Demographic Decision Making")
    with st.expander("ℹ️ How to use the Playground", expanded=True):
            st.markdown("""
            **Goal:** Understand how the AI makes decisions for a *single* specific persona.
            1. **Configure (Left):** Adjust age, income, and race to create a specific persona.
            2. **Preview (Right):** Watch how the 'Live Prompt Preview' updates. This is the exact text that send to API.
            3. **Run:** Click **Simulate** to see if this specific persona decides to report their symptoms.
            """)

    api_key = st.text_input("OpenAI API Key (Optional for Demo)", type="password")
    st.markdown(
    """
<style>
    [title="Show password text"] {
        display: none;
    }
</style>
""",
    unsafe_allow_html=True,
)
    # Split Layout
    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        st.subheader("1. Agent Configuration")
        
        # City & Context Level
        c1, c2 = st.columns(2)
        with c1: s_city = st.selectbox("City", cities)
        with c2: s_ctx_lvl = st.selectbox("Context Detail", [0, 1, 2], format_func=lambda x: ["Short", "Medium", "Long"][x])
        s_cases = st.select_slider("Reported cases", num_cases)

        st.markdown("---")
        # Demographics
        c3, c4 = st.columns(2)
        with c3:
            s_age = st.selectbox("Age", ages)
            s_race = st.selectbox("Race", races)
            s_income = st.selectbox("Income", h_incomes)
        with c4:
            s_gender = st.selectbox("Gender", genders)
            s_hisp = st.selectbox("Hispanic", hispanic_opts)
            s_edu = st.selectbox("Education", edu_levels)

        st.markdown("---")
        st.subheader("2. Experimental Factors")
        
        # Checkboxes for Variables
        use_family = st.checkbox("Include Family Influence?", value=True)
        s_fam = st.radio("Family Decision", family_d, horizontal=True) if use_family else "N/A"
        
        use_news = st.checkbox("Include News?", value=True)
        s_news_opt = st.selectbox("News Message", news_options) if use_news else "N/A"
        if s_news_opt == 'Custom (Write your own)':
            s_news_opt = st.text_area("Custom News Text")

        st.markdown("###")
        run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    with col_right:
        # Live Prompt Preview
        prompt_text = generate_prompt_text(
            s_age, s_gender, s_race, s_income, s_edu, s_cases, s_fam, s_city, s_hisp, s_news_opt, use_family, use_news, s_ctx_lvl
        )
        
        st.subheader("📝 Live Prompt Preview")
        st.caption("This is exactly what the LLM reads:")
        st.code(prompt_text, language="markdown",wrap_lines=True)
        
        # Result Area
        if run_btn:
            st.markdown("---")
            with st.spinner("Simulating..."):
                inputs = {"City": s_city, "Age": s_age, "Race": s_race, "Gender": s_gender, "Family": s_fam}
                result = run_simulation(api_key, prompt_text, inputs)
                
                if result is not None:
                    if "Yes" in str(result['Decision']):
                        st.success(f"## Decision: {result['Decision']}")
                    else:
                        st.error(f"## Decision: {result['Decision']}")
                    
                    st.write(f"**Reason:** {result['Reason']}")
                    st.metric("Reporting Confidence", f"{result['Reporting_Rate']}%")

    # --- 4. DECISION BANK (BOTTOM) ---
    st.markdown("---")
    st.subheader("📚 Decision Bank (Live Data Collection)")
    if not st.session_state['decision_bank'].empty:
        st.dataframe(st.session_state['decision_bank'], use_container_width=True)
        
        # CSV Download
        csv = st.session_state['decision_bank'].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "simulation_results.csv", "text/csv")
    else:
        st.info("Run a simulation to see data appear here.")

# ==========================================
# TAB 2: BATCH DATA AGGREGATION
# ==========================================
# ==========================================
elif st.session_state['current_page'] == 'batch':
    st.title("📊 Batch Simulation & Data Bank")
    with st.expander("ℹ️ How to use the Playground", expanded=True):
        st.caption("Step 2: Generate data at scale and create the bank for using in simulation.")
        st.markdown("""
            While Page 1 showed how **one** agent thinks, this page demonstrates how we build a **Population**.
        
        We convert each agent's demographic data into a **Vector ID** (e.g., `01010`) for using this id in the simulation. 
        """)
        
    # --- ADDED EXPLANATION: THE "MOCKUP" DISCLAIMER ---
        st.info("""
        ⚡ **Demo Mode:** To allow for instant interaction during this presentation, 
                this specific page uses a **rapid logic generator** to mimic the LLM's decision-making process. 
                In the actual research study, every single decision here is generated by an individual call to LLM model.
        """)

    # Split Screen: Left (Controls) | Right (Preview)
    col_left, col_right = st.columns([1, 1.2])

    # with col_left:
    st.subheader("1. Configure Agent Population")
    
    c1, c2 = st.columns(2)
    s_city = c1.selectbox("City", cities)
    s_ctx_lvl = c2.selectbox("Context Level", [0, 1, 2], format_func=lambda x: ["Short", "Medium", "Long"][x])

    st.markdown("---")
    
    # Demographics
    c3, c4 = st.columns(2)
    s_age = c3.selectbox("Age", ages)
    s_gender = c4.selectbox("Gender", genders)
    
    c5, c6 = st.columns(2)
    s_race = c5.selectbox("Race", races)
    s_edu = c6.selectbox("Education", edu_levels)
    
    s_inc = st.selectbox("Income", h_incomes)

    st.markdown("---")
    st.subheader("2. Experimental Variable")
    
    # The Experiment: Family Influence
    use_family = st.checkbox("Include Family Influence?", value=True)
    if use_family:
        s_fam = st.radio("Family Decision", family_d, horizontal=True)
    else:
        s_fam = "did not report" # Default fallback for vector calc
    
    # Vector Preview
    v5, v6 = get_vectors(s_age, s_gender, s_race, s_edu, s_inc, s_fam)
    st.info(f"**Target Vector IDs:**\n\nBase: `{v5}` | With Context: `{v6}`")
    
    # The Big Button
    if st.button(f"⚡ Generate Batch (10 Decisions)", type="primary", use_container_width=True):
        batch = mock_batch_generate(s_age, s_gender, s_race, s_edu, s_inc, s_fam, s_city, 10)
        st.session_state['decision_bank'] = pd.concat([st.session_state['decision_bank'], batch], ignore_index=True)
        st.success("✅ Generated 10 Decisions!")

    # with col_right:

    # --- 5. DATA AGGREGATION (The Tables) ---

    st.markdown("---")
    st.subheader("📚 Decision Bank (Live Aggregation)")

    if not st.session_state['decision_bank'].empty:
        df = st.session_state['decision_bank']
        
        # col_t1, col_t2 = st.columns(2)
        
        # TABLE 1: 5-BIT (Demographics Only)
        # with col_t1:
        st.write("#### 1. Base Demographics (5-bit ID)")
        st.caption("Aggregates all agents with this profile, regardless of Family Influence.")
        
        t1 = pd.pivot_table(df, index=['Age','Gender','Race','Education','Income','Vector_5bit'], 
                            columns='Decision', aggfunc='size', fill_value=0)
        
        # Safe column check
        for c in ['Yes', 'No']: 
            if c not in t1.columns: t1[c] = 0
            
        st.dataframe(t1[['Yes', 'No']], use_container_width=True)

        # TABLE 2: 6-BIT (With Context)
        # with col_t2:
        st.write("#### 2. Full Context (6-bit ID)")
        st.caption("Separates agents based on the 'Family Influence' variable.")
        
        t2 = pd.pivot_table(df, index=['Age','Gender','Race','Education','Income','Family_Influence','Vector_6bit'], 
                            columns='Decision', aggfunc='size', fill_value=0)
        
        for c in ['Yes', 'No']: 
            if c not in t2.columns: t2[c] = 0
            
        st.dataframe(t2[['Yes', 'No']], use_container_width=True)
        # === RAW DATA OPTION ===
        with st.expander("View Raw Data Log"):
            st.dataframe(df)
    else:
        st.info("No simulations run yet. Configure an agent on the left and click 'Generate Batch' to see the tables.")

    st.subheader("📝 Live Prompt Preview")
    
    # Generate the text to show the audience
    prompt_text = generate_prompt_text2(
        s_age, s_gender, s_race, s_inc, s_edu, s_fam, s_city, use_family, s_ctx_lvl
    )
    
    # Show it in a code block
    st.code(prompt_text, language="markdown")
    st.caption("This is the exact text the LLM reads to make its decision.")

# === PAGE 3: RESEARCH OUTCOMES ===
elif st.session_state['current_page'] == 'outcomes':
    st.title("🗺️ Simulation Research Outcomes")
    with st.expander("ℹ️ How to use the Playground", expanded=True):
        st.caption("Comparative Analysis: Baseline vs. Intervention Scenarios")
        st.markdown("""
        **Scope:** As detailed in our paper, we conducted simulations across **two major cities** to compare behavioral patterns in different urban contexts.
    
        **Instructions:**
        1. **Select City:** Use the dropdown below to switch between **Atlanta** and **San Francisco**.
        2. **Compare:** We analyze three scenarios (Baseline, Family Influence, News) to see which intervention is most effective in each city.
       
        We will use the 3 generated files from the simulation
                    (AgentCharacteristicsTable.tsv, DiseaseReports.tsv and patterns_of_life.log) to create these plot.
                
        """)
    if not HAS_GEOPANDAS:
        st.error("⚠️ GeoPandas not detected. Maps cannot be rendered.")

    # 1. Configuration (Top Bar)
    with st.expander("📂 Data Configuration", expanded=True):
        c1, c2 = st.columns([1, 2])
        selected_city = c1.selectbox("Select City Dataset", ["atlanta", "san_francisco"], 
                                    format_func=lambda x: x.replace('_', ' ').title())

        # Dynamic Path Construction
        prefix = "atl" if selected_city == "atlanta" else "san"
        map_folder = "atlanta" if selected_city == "atlanta" else "san"

        # --- YOUR ORIGINAL CONFIGURATION ---
        scenarios_config = [
            {'name': '1. Baseline',         'folder': f'{prefix}-llm_test40day_V1-5_S1'},
            {'name': '2. Family Influence', 'folder': f'{prefix}-llm_test40day_V1-5_S2'},
            {'name': '3. News',             'folder': f'{prefix}-llm_test40day_V1-5_S3'} 
        ]
        # -----------------------------------

        if st.button("🔄 Load & Compare All Scenarios", type="primary"):
            # Initialize session state
            st.session_state['comparison_data'] = {}
            st.session_state['comparison_city'] = selected_city
            
            # 1. Create Cache Directory
            cache_dir = "processed_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            # 2. Progress Bar
            progress_bar = st.progress(0)
            
            # 3. Loop through scenarios with Caching Logic
            for i, sc in enumerate(scenarios_config):
                # Construct full path to raw data
                full_path = f"{root_path}/{sc['folder']}"
                
                # Construct unique filename for the cache (e.g., "atlanta_1_Baseline.pkl")
                safe_name = sc['name'].replace(' ', '_').replace('.', '')
                cache_path = f"{cache_dir}/{selected_city}_{safe_name}.pkl"
                
                geo_df = pd.DataFrame() # Reset
                
                # --- A. TRY LOADING FROM CACHE ---
                if os.path.exists(cache_path):
                    try:
                        # Fast load
                        geo_df = pd.read_pickle(cache_path)
                        st.toast(f"Loaded {sc['name']} from cache", icon="⚡")
                    except Exception as e:
                        st.warning(f"Cache failed for {sc['name']}, reprocessing...")
                
                # --- B. IF NO CACHE, PROCESS RAW FILES ---
                if geo_df.empty:
                    try:
                        # Slow processing (Your function)
                        _, geo_df = process_raw_data_files_p(map_folder, full_path)
                        
                        # Save to cache for next time
                        if not geo_df.empty:
                            geo_df.to_pickle(cache_path)
                            st.toast(f"Processed and cached {sc['name']}", icon="💾")
                    except Exception as e:
                        st.error(f"Error processing {sc['name']}: {e}")

                # --- C. STORE IN SESSION STATE ---
                if not geo_df.empty:
                    geo_df['Scenario'] = sc['name']
                    st.session_state['comparison_data'][sc['name']] = geo_df
                
                # Update progress
                progress_bar.progress(int((i + 1) / len(scenarios_config) * 100))
            
            st.rerun()

    # -----------------------------------------------------------------------------
    # 2. Visualization Area (Rest of your code)
    # -----------------------------------------------------------------------------
    if 'comparison_data' in st.session_state and st.session_state['comparison_data']:
        data_map = st.session_state['comparison_data']
        
        st.subheader("📍 Geographic Distribution of Reporting Rates")
        st.caption("""
        **How to read these maps:** Each shape represents a neighborhood (Census Tract). 
        - **Darker Blue** = High Reporting Rate (Most people in this area decided to report).
        - **Lighter Blue** = Low Reporting Rate (Most people hid their symptoms).
        
        Compare the maps side-by-side to see if the maps get "Darker" (better reporting) in the intervention scenarios.
        """)
    #     c1, c2 = st.columns([1, 2])
    #     selected_city = c1.selectbox("Select City Dataset", ["atlanta", "san_francisco"], format_func=lambda x: x.replace('_', ' ').title())
        
    #     # Dynamic Path Construction based on City
    #     # Assumes folder naming convention: "atl-llm..." or "sf-llm..."
    #     prefix = "atl" if selected_city == "atlanta" else "san"
    #     map_folder = "atlanta" if selected_city == "atlanta" else "san"
        
    #     scenarios_config = [
    #         {'name': '1. Baseline',       'folder': f'{prefix}-llm_test40day_V1-5_S1'},
    #         {'name': '2. Family Influence', 'folder': f'{prefix}-llm_test40day_V1-5_S2'},
    #         {'name': '3. News',   'folder': f'{prefix}-llm_test40day_V1-5_S3'} 
    #     ]
        
    #     if st.button("🔄 Load & Compare All Scenarios", type="primary"):
    #         st.session_state['comparison_data'] = {}
    #         st.session_state['comparison_city'] = selected_city
            
    #         # Progress bar for loading 3 heavy files
    #         progress_bar = st.progress(0)
            
    #         for i, sc in enumerate(scenarios_config):
    #             full_path = f"{root_path}/{sc['folder']}"
                
    #             # Call your processing function
    #             summary, geo_df = process_raw_data_files_p(map_folder, full_path)
                
    #             if not geo_df.empty:
    #                 geo_df['Scenario'] = sc['name']
    #                 st.session_state['comparison_data'][sc['name']] = geo_df
                
    #             progress_bar.progress((i + 1) * 33)
            
    #         st.rerun()

    # # 2. Visualization Area
    # if 'comparison_data' in st.session_state and st.session_state['comparison_data']:
    #     data_map = st.session_state['comparison_data']
        
    #     # --- A. SIDE-BY-SIDE MAPS ---
    #     st.subheader("📍 Geographic Distribution of Reporting Rates")
        
        # Calculate Global Min/Max for unified color scale (Crucial for comparison!)
        all_dfs = list(data_map.values())
        vmin = min([df['% Component'].min() for df in all_dfs])
        vmax = max([df['% Component'].max() for df in all_dfs])
        
        # Create 3 columns
        cols = st.columns([1, 1, 1, 2])
        scenario_names = list(data_map.keys())
        flag_legend = False
        for i, col in enumerate(cols):
            if i == len(scenario_names) - 1:
                flag_legend = True
            if i < len(scenario_names):
                name = scenario_names[i]
                gdf = data_map[name]
                
                with col:
                    st.write(f"**{name}**")
                    if HAS_GEOPANDAS:
                        fig, ax = plt.subplots(figsize=(2,2))
                        gdf.plot(column='% Component', 
                                 cmap='Blues', 
                                 linewidth=0.5, 
                                 ax=ax, 
                                 edgecolor='0.9', 
                                 vmin=vmin, vmax=vmax) # Unified Scale
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.write("Map unavailable")
        if HAS_GEOPANDAS:
            # Add a little vertical space
            st.markdown("") 
            cbar_col, _ = st.columns([3, 2])
            with cbar_col:
                # Ultra thin height (0.2 inch)
                fig_cbar, ax_cbar = plt.subplots(figsize=(5, 0.2)) 
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
                sm.set_array([])
                
                cbar = plt.colorbar(sm, cax=ax_cbar, orientation='horizontal')
                cbar.ax.tick_params(labelsize=7) 
                cbar.set_label('Reporting Rate', fontsize=7) 
                st.pyplot(fig_cbar, use_container_width=False)

        # --- B. STATISTICAL COMPARISON ---
        st.markdown("---")
        st.subheader("📊 Statistical Impact")
        st.markdown("""
        **Distribution Analysis (Boxplot):**
        The maps show *where* things happened, but this chart shows *how much* behaviour changed overall.
       """)
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Constrain width: Chart gets 50%, Stats gets 30%, Empty 20%
        c_chart, c_stats, _ = st.columns([2, 1, 1]) 
        
        with c_chart:
            st.caption("Distribution Analysis (Boxplot)")
            
            # SHORT HEIGHT: (5 wide, 2 high)
            fig2, ax2 = plt.subplots(figsize=(5, 2)) 
            
            sns.boxplot(x='Scenario', y='% Component', data=combined_df, palette="Oranges", ax=ax2, linewidth=0.8, fliersize=2)
            
            # Minimalist styling
            ax2.set_ylabel("Reporting Rate", fontsize=7)
            ax2.set_xlabel("", fontsize=0)
            ax2.tick_params(labelsize=7)
            sns.despine(left=True)
            ax2.grid(axis='y', linestyle='--', alpha=0.3)
            
            st.pyplot(fig2, use_container_width=False)
            
        with c_stats:
            st.caption("Average Rate Summary")
            stats = combined_df.groupby('Scenario')[['% Component']].mean().sort_values('% Component')
            # Show simple table
            st.dataframe(stats.style.format("{:.1%}"), use_container_width=True, height=150)
            
            if '1. Baseline' in stats.index and '3. Full Context' in stats.index:
                base = stats.loc['1. Baseline', '% Component']
                final = stats.loc['3. Full Context', '% Component']
                lift = (final - base) / base
                st.markdown(f"**Improvement:** :green[+{lift:.1%}]")

    else:
        st.info(f"Ready to load data. Please ensure path is correct: `{root_path}`")
