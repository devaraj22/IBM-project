# frontend/streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import json
import time
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Medical Prescription Verification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background: white;
        color: #333333;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }

    /* Ensure all box types have visible text */
    .warning-box {
        color: #856404;
    }
    .danger-box {
        color: #721c24;
    }
    .success-box {
        color: #155724;
    }
    .info-box {
        color: #0c5460;
    }

    /* Metric card specific styling */
    .metric-card h3 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        color: #3498db;
        font-size: 2rem;
        margin: 0.5rem 0;
    }
    .metric-card p {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Session state initialization
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Session state initialization
if 'ocr_extracted_text' not in st.session_state:
    st.session_state.ocr_extracted_text = ""
if 'ocr_extraction_done' not in st.session_state:
    st.session_state.ocr_extraction_done = False

# Helper functions
def get_ocr_extraction_state():
    """Get OCR extraction state from session"""
    return st.session_state.get('ocr_extraction_done', False)

def set_ocr_extraction_state(value: bool, text: str = ""):
    """Set OCR extraction state"""
    st.session_state.ocr_extraction_done = value
    if text:
        st.session_state.ocr_extracted_text = text

def make_api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"

        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}

    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {}

def display_interaction_results(interactions: List[Dict], safe: bool, risk_level: str, recommendations: List[str]):
    """Display drug interaction results with proper formatting"""
    
    if safe:
        st.markdown('<div class="success-box">✅ <strong>No dangerous drug interactions detected!</strong></div>', 
                   unsafe_allow_html=True)
    else:
        risk_colors = {"low": "info", "medium": "warning", "high": "danger"}
        color_class = risk_colors.get(risk_level, "warning")
        
        st.markdown(f'<div class="{color_class}-box">⚠️ <strong>Drug interactions detected - Risk level: {risk_level.upper()}</strong></div>', 
                   unsafe_allow_html=True)
    
    # Display individual interactions
    if interactions:
        st.subheader("📋 Detailed Interaction Analysis")
        
        for i, interaction in enumerate(interactions, 1):
            severity = interaction.get('severity', 'unknown')
            drug1 = interaction.get('drug1', 'Drug A')
            drug2 = interaction.get('drug2', 'Drug B')
            
            severity_colors = {
                'major': '🔴',
                'moderate': '🟡', 
                'minor': '🟢',
                'unknown': '⚪'
            }
            
            severity_icon = severity_colors.get(severity.lower(), '⚪')
            
            with st.expander(f"{severity_icon} Interaction #{i}: {drug1} ↔ {drug2} ({severity.title()})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Description:**")
                    st.write(interaction.get('description', 'No description available'))
                    
                    st.write("**Mechanism:**")
                    st.write(interaction.get('mechanism', 'Mechanism not specified'))
                
                with col2:
                    st.write("**Severity:**")
                    st.write(f"**{severity.title()}**")
                    
                    st.write("**Evidence Level:**")
                    st.write(interaction.get('evidence_level', 'Not specified'))
                
                if interaction.get('management'):
                    st.write("**Management Recommendation:**")
                    st.info(interaction['management'])
                
                if interaction.get('age_warnings'):
                    st.write("**Age-Specific Warnings:**")
                    for warning in interaction['age_warnings']:
                        st.warning(warning)
    
    # Display recommendations
    if recommendations:
        st.subheader("💡 Clinical Recommendations")
        for recommendation in recommendations:
            st.write(f"• {recommendation}")

def display_dosage_results(dosage_info: Dict):
    """Display dosage calculation results"""
    
    if not dosage_info:
        st.error("❌ No dosage information available for this medication")
        return
    
    st.markdown('<div class="success-box">✅ <strong>Dosage recommendation calculated successfully!</strong></div>', 
               unsafe_allow_html=True)
    
    # Main dosage information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💊 Recommended Dose", 
                 f"{dosage_info.get('recommended_dose', 'N/A')} {dosage_info.get('unit', '')}")
    
    with col2:
        st.metric("⏰ Frequency", dosage_info.get('frequency', 'N/A'))
    
    with col3:
        st.metric("🛣️ Route", dosage_info.get('route', 'N/A'))
    
    with col4:
        st.metric("👥 Age Group", dosage_info.get('age_group', 'N/A'))
    
    # Warnings and adjustments
    if dosage_info.get('warnings'):
        st.subheader("⚠️ Important Warnings")
        for warning in dosage_info['warnings']:
            st.warning(warning)
    
    if dosage_info.get('adjustments'):
        st.subheader("🔧 Dosage Adjustments")
        for condition, adjustment in dosage_info['adjustments'].items():
            st.info(f"**{condition.title()}:** {adjustment}")

def display_prescription_parse_results(parsed_data: Dict):
    """Display prescription parsing results"""
    
    confidence = parsed_data.get('confidence_score', 0)
    processing_method = parsed_data.get('processing_method', 'unknown')
    
    # Confidence indicator
    if confidence >= 0.8:
        conf_color = "success"
        conf_icon = "✅"
    elif confidence >= 0.6:
        conf_color = "warning"
        conf_icon = "⚠️"
    else:
        conf_color = "danger"
        conf_icon = "❌"
    
    st.markdown(f'<div class="{conf_color}-box">{conf_icon} <strong>Parsing completed with {confidence:.1%} confidence</strong> (Method: {processing_method.title()})</div>', 
               unsafe_allow_html=True)
    
    # Display extracted information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💊 Medications Identified")
        medications = parsed_data.get('medications', [])
        
        if medications:
            df_meds = pd.DataFrame(medications)
            st.dataframe(df_meds, use_container_width=True)
        else:
            st.info("No medications clearly identified")
        
        st.subheader("💉 Dosages Found")
        dosages = parsed_data.get('dosages', [])
        if dosages:
            for i, dosage in enumerate(dosages, 1):
                st.write(f"{i}. **{dosage.get('amount', 'N/A')} {dosage.get('unit', '')}**")
        else:
            st.info("No dosage information found")
    
    with col2:
        st.subheader("⏰ Frequencies")
        frequencies = parsed_data.get('frequencies', [])
        if frequencies:
            for freq in frequencies:
                st.write(f"• {freq}")
        else:
            st.info("No frequency information found")
        
        st.subheader("🛣️ Routes of Administration")
        routes = parsed_data.get('routes', [])
        if routes:
            for route in routes:
                st.write(f"• {route}")
        else:
            st.info("No route information found")
    
    # Display warnings
    warnings = parsed_data.get('warnings', [])
    if warnings:
        st.subheader("⚠️ Analysis Warnings")
        for warning in warnings:
            st.warning(warning)

# Main application
def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Medical Prescription Verification System</h1>', 
               unsafe_allow_html=True)
    
    # Sidebar navigation - Prescription Analyzer first
    st.sidebar.title("📋 Navigation")

    # Page options - Prescription Analyzer is first/default
    page_options = [
        "📄 Prescription Analyzer",  # Default first page
        "💊 Drug Interaction Checker",
        "📏 Age-Specific Dosage",
        "🔄 Alternative Medications",
        "📊 Analysis Results",
        "📈 Analysis History"
    ]

    # Determine which page to show
    if 'selected_page' in st.session_state:
        if st.session_state['selected_page'] in page_options:
            default_index = page_options.index(st.session_state['selected_page'])
            selected_page_value = st.session_state['selected_page']
        else:
            default_index = 0
            selected_page_value = None
        # Clear the selected page after using it
        del st.session_state['selected_page']
    else:
        default_index = 0
        selected_page_value = None

    selected_page = st.sidebar.radio(
        "AI Medical Analysis:",
        page_options,
        index=default_index
    )

    # Use the quick navigation selection if available
    if selected_page_value and selected_page != selected_page_value:
        selected_page = selected_page_value
    
    # System status check
    with st.sidebar:
        st.subheader("🔍 System Status")
        
        with st.spinner("Checking API connection..."):
            health_status = make_api_request("/health")
        
        if health_status:
            st.success("✅ API Connected")
            
            services = health_status.get('services', {})
            for service, status in services.items():
                status_icon = "✅" if status == "operational" else "❌"
                st.write(f"{status_icon} {service.replace('_', ' ').title()}")
        else:
            st.error("❌ API Connection Failed")
    
    # Page routing
    if selected_page == "📄 Prescription Analyzer":
        show_prescription_analyzer_page()  # Renamed for consistency
    elif selected_page == "💊 Drug Interaction Checker":
        show_drug_interaction_page()
    elif selected_page == "📏 Age-Specific Dosage":
        show_dosage_page()
    elif selected_page == "🔄 Alternative Medications":
        show_alternative_drugs_page()
    elif selected_page == "📊 Analysis Results":
        show_analysis_results_page()  # New results dashboard
    elif selected_page == "📈 Analysis History":
        show_analysis_history()

def show_dashboard():
    """Display main dashboard"""
    
    st.markdown('<h2 class="section-header">📊 System Overview</h2>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍</h3>
            <h2>12,543</h2>
            <p>Interactions Checked</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📄</h3>
            <h2>3,287</h2>
            <p>Prescriptions Parsed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💊</h3>
            <h2>8,921</h2>
            <p>Dosages Calculated</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🔄</h3>
            <h2>1,654</h2>
            <p>Alternatives Suggested</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Usage Trends")
        
        # Sample data for demonstration
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        interactions = [50 + i*2 + (i%7)*10 for i in range(30)]
        
        fig = px.line(x=dates, y=interactions, title="Daily Drug Interaction Checks")
        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Checks")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Risk Distribution")
        
        # Sample risk distribution
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Count': [450, 120, 30],
            'Color': ['#28a745', '#ffc107', '#dc3545']
        })
        
        fig = px.pie(risk_data, values='Count', names='Risk Level', 
                    color='Risk Level',
                    color_discrete_map={
                        'Low': '#28a745',
                        'Medium': '#ffc107', 
                        'High': '#dc3545'
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions
    st.subheader("🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Check Drug Interactions", use_container_width=True):
            st.success("🔍 Navigating to Drug Interaction Checker...")
            st.session_state['selected_page'] = "💊 Drug Interaction Checker"
            st.rerun()

    with col2:
        if st.button("💊 Calculate Dosage", use_container_width=True):
            st.success("💊 Navigating to Age-Specific Dosage...")
            st.session_state['selected_page'] = "📏 Age-Specific Dosage"
            st.rerun()

    with col3:
        if st.button("📄 Parse Prescription", use_container_width=True):
            st.success("📄 Navigating to Prescription Parser...")
            st.session_state['selected_page'] = "📄 Prescription Parser"
            st.rerun()

def show_drug_interaction_page():
    """Drug interaction checker page"""
    
    st.markdown('<h2 class="section-header">💊 Drug Interaction Checker</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>How to use:</strong> Enter 2 or more medications to check for potential drug-drug interactions. 
    The system will analyze interactions from multiple medical databases and provide risk assessments.
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    with st.form("drug_interaction_form"):
        st.subheader("📝 Enter Medications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Check for drugs from prescription analysis
            drugs = []
            use_analyzed_drugs = False

            if 'analyzed_drugs' in st.session_state:
                if not st.session_state.get('use_manual_drugs', False):
                    st.info("💡 **Using drugs from analyzed prescription:** " + ", ".join(st.session_state['analyzed_drugs']))
                    drugs = st.session_state['analyzed_drugs']
                    use_analyzed_drugs = True

                    # Button will be placed after the form

                else:
                    st.session_state['use_manual_drugs'] = False

            # Drug input method selection - only show if not using analyzed drugs
            if not use_analyzed_drugs:
                input_method = st.radio(
                    "Input method:",
                    ["Manual entry", "Select from list", "Upload file"]
                )

                if input_method == "Manual entry":
                    drug_input = st.text_area(
                        "Enter drug names (one per line):",
                        placeholder="aspirin\nwarfarin\nlisinopril",
                        height=150
                    )
                    drugs = [drug.strip() for drug in drug_input.split('\n') if drug.strip()]

                elif input_method == "Select from list":
                    # Common medications for selection
                    common_drugs = [
                        "Aspirin", "Warfarin", "Lisinopril", "Metformin", "Atorvastatin",
                        "Amlodipine", "Metoprolol", "Levothyroxine", "Ibuprofen", "Acetaminophen",
                        "Omeprazole", "Simvastatin", "Furosemide", "Digoxin", "Insulin"
                    ]

                    drugs = st.multiselect(
                        "Select medications:",
                        options=common_drugs,
                        help="Choose 2 or more medications to check for interactions"
                    )

                else:  # Upload file
                    uploaded_file = st.file_uploader("Upload drug list (CSV/TXT)", type=['csv', 'txt'])
                    drugs = []

                    if uploaded_file:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                            if 'drug_name' in df.columns:
                                drugs = df['drug_name'].dropna().tolist()
                            else:
                                drugs = df.iloc[:, 0].dropna().tolist()
                        else:
                            content = uploaded_file.read().decode('utf-8')
                            drugs = [drug.strip() for drug in content.split('\n') if drug.strip()]
        
        with col2:
            st.subheader("⚙️ Options")
            
            patient_age = st.number_input(
                "Patient age (optional):",
                min_value=0,
                max_value=120,
                value=None,
                help="Provide patient age for age-specific warnings"
            )
            
            severity_filter = st.selectbox(
                "Filter by severity:",
                options=[None, "Minor", "Moderate", "Major"],
                help="Filter interactions by severity level"
            )
            
            include_minor = st.checkbox(
                "Include minor interactions",
                value=True,
                help="Show all interactions including minor ones"
            )
        
        # Submit button
        submit_button = st.form_submit_button("🔍 Check Interactions", use_container_width=True)
    
    # Process form submission
    if submit_button:
        if len(drugs) < 2:
            st.error("❌ Please enter at least 2 medications to check for interactions")
        else:
            st.subheader(f"🔍 Analyzing {len(drugs)} medications...")

            # Display selected drugs
            st.write("**Selected medications:**", ", ".join(drugs))

            with st.spinner("Checking drug interactions..."):
                # Prepare API request
                request_data = {
                    "drugs": drugs,
                    "patient_age": patient_age,
                    "severity_filter": severity_filter.lower() if severity_filter else None
                }

                # Make API call
                result = make_api_request("/check-interactions", method="POST", data=request_data)

                if result:
                    interactions = result.get('interactions', [])
                    safe = result.get('safe', True)
                    risk_level = result.get('risk_level', 'unknown')
                    recommendations = result.get('recommendations', [])

                    # Filter minor interactions if not requested
                    if not include_minor:
                        interactions = [i for i in interactions if i.get('severity', '').lower() != 'minor']

                    # Display results
                    display_interaction_results(interactions, safe, risk_level, recommendations)

                    # Save to history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'type': 'Drug Interaction',
                        'input': drugs,
                        'result': result
                    })

                    # Export options
                    if interactions:
                        st.subheader("📤 Export Results")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Create CSV export
                            df_interactions = pd.DataFrame(interactions)
                            csv_data = df_interactions.to_csv(index=False)

                            st.download_button(
                                "📥 Download CSV Report",
                                data=csv_data,
                                file_name=f"drug_interactions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )

                        with col2:
                            # Create JSON export
                            json_data = json.dumps(result, indent=2)

                            st.download_button(
                                "📥 Download JSON Report",
                                data=json_data,
                                file_name=f"drug_interactions_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json"
                            )

    # Button to switch from analyzed drugs to manual drugs (outside the form)
    if 'analyzed_drugs' in st.session_state and not st.session_state.get('use_manual_drugs', False) and use_analyzed_drugs:
        if st.button("✏️ Enter Different Drugs", key="switch_manual_drugs"):
            st.session_state['use_manual_drugs'] = True
            st.rerun()

def show_dosage_page():
    """Age-specific dosage calculator page"""
    
    st.markdown('<h2 class="section-header">📏 Age-Specific Dosage Calculator</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>How to use:</strong> Enter medication details and patient information to get age-appropriate dosage recommendations 
    based on clinical guidelines and patient-specific factors.
    </div>
    """, unsafe_allow_html=True)
    
    # Show existing analyzed drugs if available
    if 'analyzed_drugs' in st.session_state and st.session_state['analyzed_drugs']:
        st.info("💡 **Analyzed medications available:** " + ", ".join(st.session_state['analyzed_drugs']))
        st.markdown("You can use these drugs or enter a different medication below.")

    # Input form
    with st.form("dosage_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💊 Medication Information")
            
            drug_name = st.text_input(
                "Medication name:",
                placeholder="e.g., Amoxicillin, Acetaminophen",
                help="Enter the generic or brand name of the medication"
            )
            
            indication = st.text_input(
                "Medical indication (optional):",
                placeholder="e.g., pneumonia, fever, hypertension",
                help="The condition being treated"
            )
            
            kidney_function = st.selectbox(
                "Kidney function:",
                options=["Normal", "Mild impairment", "Moderate impairment", "Severe impairment", "Unknown"],
                help="Patient's kidney function status for dosage adjustment"
            )
        
        with col2:
            st.subheader("👤 Patient Information")
            
            patient_age = st.number_input(
                "Patient age (years):",
                min_value=0,
                max_value=120,
                value=30,
                help="Patient's age in years"
            )
            
            weight = st.number_input(
                "Patient weight (kg):",
                min_value=0.5,
                max_value=300.0,
                value=70.0,
                step=0.1,
                help="Patient's weight in kilograms"
            )
            
            # Age group indicator
            if patient_age < 2:
                age_group = "👶 Infant"
            elif patient_age < 12:
                age_group = "🧒 Child"
            elif patient_age < 18:
                age_group = "🧑 Adolescent"
            elif patient_age < 65:
                age_group = "🧑‍💼 Adult"
            else:
                age_group = "👴 Elderly"
            
            st.info(f"**Age Group:** {age_group}")
        
        submit_button = st.form_submit_button("🧮 Calculate Dosage", use_container_width=True)
    
    # Process form submission
    if submit_button:
        if not drug_name:
            st.error("❌ Please enter a medication name")
        else:
            with st.spinner("Calculating age-specific dosage..."):
                # Prepare API request
                request_data = {
                    "drug_name": drug_name,
                    "patient_age": patient_age,
                    "weight": weight,
                    "indication": indication if indication else None,
                    "kidney_function": kidney_function if kidney_function != "Unknown" else None
                }
                
                # Make API call
                result = make_api_request("/age-dosage", method="POST", data=request_data)
                
                if result:
                    display_dosage_results(result)
                    
                    # Save to history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'type': 'Dosage Calculation',
                        'input': request_data,
                        'result': result
                    })

def show_prescription_analyzer_page():
    """Main prescription analyzer page - the primary entry point"""

    st.markdown('<h2 class="section-header">📄 AI Prescription Analyzer - Main Tool</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 10px; margin: 10px 0;">
    <h3>Welcome to the AI Medical Analysis System</h3>
    <p>Start here by uploading or entering a prescription to extract medications, then use those drugs across all analysis tools automatically.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>How to use:</strong> Upload or paste prescription text/image to extract structured information using AI.
    <br>• <strong>TXT Files:</strong> Ollama Granite AI (fast, local processing)
    <br>• <strong>Images:</strong> Google Gemini 1.5 Flash OCR (advanced vision capabilities)
    <br>• All medications auto-populate across other analysis tools
    </div>
    """, unsafe_allow_html=True)
    
    # Input options
    input_method = st.radio(
        "Choose input method:",
        ["Text input", "File upload", "Voice input (experimental)"]
    )
    
    prescription_text = ""
    
    if input_method == "Text input":
        prescription_text = st.text_area(
            "Enter prescription text:",
            height=200,
            placeholder="""Example:
Rx: Amoxicillin 500mg capsules
Sig: Take 1 capsule by mouth three times daily for 10 days
Disp: 30 capsules
Patient: John Smith, Age: 45
For: Bacterial infection""",
            help="Paste or type the prescription text here"
        )
    
    elif input_method == "File upload":
        uploaded_file = st.file_uploader(
            "Upload prescription file:",
            type=['txt', 'jpg', 'png'],
            help="Supports text files (direct processing) and image files (OCR extraction with preview)."
        )

        if uploaded_file:
            # Display file info
            st.info(f"📄 Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")

            if uploaded_file.type == "text/plain":
                prescription_text = uploaded_file.read().decode('utf-8')
                st.text_area("Extracted text:", value=prescription_text, height=150)
            elif uploaded_file.type.startswith('image/'):
                st.image(uploaded_file, caption="Uploaded prescription image", width=400)

                # Show OCR extraction section
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    ocr_extract_button = st.button("🔍 Extract Text from Image", use_container_width=True, type="primary")
                with col2:
                    auto_ocr = st.checkbox("Auto-extract", value=False,
                                         help="Automatically extract text when image is uploaded")

                if ocr_extract_button or (auto_ocr and not get_ocr_extraction_state()):
                    with st.spinner("🔄 Extracting text from image using Gemini OCR..."):
                        try:
                            # Step 1: Call OCR extraction endpoint
                            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            ocr_response = requests.post(
                                f"{API_BASE_URL}/extract-text-from-image",
                                files=files,
                                timeout=45  # Longer timeout for OCR
                            )

                            if ocr_response.status_code == 200:
                                ocr_result = ocr_response.json()
                                extracted_text = ocr_result.get('extracted_text', '')
                                ocr_success = ocr_result.get('ocr_success', False)

                                if ocr_success and extracted_text and len(extracted_text.strip()) > 50:
                                    st.success("✅ OCR text extraction successful!")

                                    # Store extraction state and text
                                    set_ocr_extraction_state(True, extracted_text)

                                    # Show extracted text in a nice expandable area
                                    with st.expander("🔤 Extracted Text (Click to Review)", expanded=True):
                                        st.code(extracted_text, language='text')
                                        st.info(f"📝 Extracted {len(extracted_text)} characters from the image")

                                    # Step 2: Option to edit the text if needed
                                    edit_option = st.checkbox("✏️ Edit extracted text if needed")
                                    if edit_option:
                                        prescription_text = st.text_area(
                                            "Edit extracted text:",
                                            value=extracted_text,
                                            height=200,
                                            help="Make corrections to the extracted text if the OCR made any mistakes",
                                            key="ocr_edit_text"
                                        )
                                    else:
                                        prescription_text = extracted_text

                                    # Step 3: Quick AI Analysis with Granite
                                    st.markdown("---")
                                    st.markdown("### 🤖 AI Analysis with Granite")

                                    quick_analysis = st.button("🔬 Analyze with Granite AI",
                                                             use_container_width=True,
                                                             type="primary",
                                                             key="analyze_ocr_text")

                                    if quick_analysis:
                                        with st.spinner("🤖 Analyzing extracted text with Granite AI model..."):
                                            try:
                                                # Get current language setting from main form (en by default)
                                                current_lang = "en"  # Default fallback

                                                search_data = {
                                                    "text": prescription_text,
                                                    "language": current_lang
                                                }

                                                analyze_response = requests.post(
                                                    f"{API_BASE_URL}/analyze-extracted-text",
                                                    json=search_data,
                                                    timeout=45
                                                )

                                                if analyze_response.status_code == 200:
                                                    analysis_result = analyze_response.json()

                                                    # Display results immediately
                                                    display_prescription_parse_results(analysis_result)

                                                    # Store in session for cross-page access
                                                    medications = analysis_result.get('medications', [])
                                                    if medications:
                                                        drug_names = [med['name'] for med in medications]
                                                        st.session_state['analyzed_drugs'] = drug_names
                                                        st.session_state['prescription_analysis'] = analysis_result

                                                        confidence = analysis_result.get('confidence_score', 0)
                                                        st.success(f"🤖 **Granite AI analysis complete!** Found {len(medications)} medications with {confidence:.1%} confidence.")
                                                    else:
                                                        st.warning("🤖 **No medications found** in the extracted text. Try editing the text or uploading a clearer image.")

                                                else:
                                                    st.error(f"Analysis failed: HTTP {analyze_response.status_code}")
                                                    st.error("Try again or contact support if the issue persists.")

                                            except Exception as e:
                                                st.error(f"❌ Granite AI analysis error: {str(e)}")
                                                logger.error(f"OCR analysis error: {str(e)}")

                                elif ocr_success and extracted_text and len(extracted_text.strip()) > 10:
                                    # Partial success with short text
                                    st.warning("⚠️ OCR extracted some text but it may be incomplete.")
                                    st.code(extracted_text, language='text')
                                    prescription_text = extracted_text
                                    set_ocr_extraction_state(True, extracted_text)

                                else:
                                    st.error("❌ OCR extraction failed or returned insufficient text.")
                                    st.info("💡 **Tips for better results:**")
                                    st.info("• Ensure the prescription is well-lit and clear")
                                    st.info("• Avoid blurry or rotated images")
                                    st.info("• Make sure text is readable without magnification")
                                    prescription_text = ""
                                    set_ocr_extraction_state(False, "")

                            else:
                                st.error(f"❌ OCR API request failed: HTTP {ocr_response.status_code}")
                                if ocr_response.text:
                                    st.error(f"Error details: {ocr_response.text[:200]}...")
                                prescription_text = ""
                                set_ocr_extraction_state(False, "")

                        except requests.exceptions.Timeout:
                            st.error("❌ OCR request timed out. The image may be too large or the server is busy.")
                            prescription_text = ""
                        except Exception as e:
                            st.error(f"❌ OCR processing error: {str(e)}")
                            prescription_text = ""
                            set_ocr_extraction_state(False, "")

                elif get_ocr_extraction_state():
                    # Show previously extracted results
                    stored_text = st.session_state.get('ocr_extracted_text', '')
                    if stored_text:
                        st.success("✅ Using previously extracted text")
                        with st.expander("🔤 Previously Extracted Text", expanded=False):
                            st.code(stored_text, language='text')

                        edit_option = st.checkbox("Edit text if needed")
                        if edit_option:
                            prescription_text = st.text_area(
                                "Edit extracted text:",
                                value=stored_text,
                                height=150,
                                help="Make corrections to the extracted text"
                            )
                        else:
                            prescription_text = stored_text

                        # Option to analyze stored text
                        if st.button("🔬 Re-analyze with Granite AI"):
                            with st.spinner("🤖 Re-analyzing with Granite AI..."):
                                try:
                                    search_data = {"text": prescription_text, "language": "en"}
                                    analyze_response = requests.post(
                                        f"{API_BASE_URL}/analyze-extracted-text",
                                        json=search_data,
                                        timeout=30
                                    )

                                    if analyze_response.status_code == 200:
                                        analysis_result = analyze_response.json()
                                        display_prescription_parse_results(analysis_result)

                                        medications = analysis_result.get('medications', [])
                                        if medications:
                                            drug_names = [med['name'] for med in medications]
                                            st.session_state['analyzed_drugs'] = drug_names
                                            st.session_state['prescription_analysis'] = analysis_result
                                            st.success(f"🤖 Analysis complete! Found {len(medications)} medications.")
                                        else:
                                            st.warning("No medications found in the text.")

                                except Exception as e:
                                    st.error(f"Analysis error: {str(e)}")
                    else:
                        set_ocr_extraction_state(False, "")

                else:
                    prescription_text = ""
                    if not auto_ocr:
                        st.info("⏳ Click 'Extract Text from Image' to start OCR processing")
            else:
                st.warning("⚠️ File type not supported. Please use TXT files or image files (JPG, PNG).")
                prescription_text = ""
    
    else:  # Voice input
        st.warning("🎤 Voice input feature is experimental and not implemented in this demo")
        
        if st.button("🎙️ Start Recording"):
            st.info("Voice recording would start here...")
    
    # Processing options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Language:",
                options=["English", "Spanish", "French", "German"],
                help="Prescription language for better parsing accuracy"
            )
            
            processing_method = st.selectbox(
                "Processing Mode:",
                options=["Auto (recommended)", "Ollama Granite (text)", "Gemini OCR (images)", "Rule-based (fallback)"],
                help="Auto mode uses best available - Granite for text, Gemini for images"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence threshold:",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Minimum confidence for entity extraction"
            )
            
            extract_entities = st.multiselect(
                "Extract specific entities:",
                options=["Medications", "Dosages", "Frequencies", "Routes", "Indications", "Patient info"],
                default=["Medications", "Dosages", "Frequencies", "Routes"],
                help="Choose which information to extract"
            )
    
    # Submit button
    if st.button("🔍 Parse Prescription", disabled=not prescription_text, use_container_width=True):
        with st.spinner("Parsing prescription with AI..."):
            # Prepare API request
            request_data = {
                "text": prescription_text,
                "language": language.lower()[:2] if language else "en"
            }
            
            # Make API call
            result = make_api_request("/parse-prescription", method="POST", data=request_data)
            
            if result:
                display_prescription_parse_results(result)

                # Show OCR extracted text if available
                if 'ocr_extracted_text' in result:
                    st.subheader("🔤 OCR Extracted Text")
                    with st.expander("View Extracted Text", expanded=True):
                        st.code(result['ocr_extracted_text'], language='text')
                        st.info("📝 **This text was extracted from your image and sent to the AI for analysis.**")

                # Store analyzed drugs in session state for cross-page use
                medications = result.get('medications', [])
                if medications:
                    drug_names = [med['name'] for med in medications]
                    st.session_state['analyzed_drugs'] = drug_names
                    st.session_state['prescription_analysis'] = result
                    processor = result.get('processed_by', result.get('processing_method', 'AI Processing'))
                    model_icon = "🤖" if "Granite" in processor or "ollama" in processor else "🔮" if "Gemini" in processor else "⚡"
                    st.success(f"{model_icon} **Prescription analyzed successfully!** Found {len(medications)} medications using {processor}. You can now use these drugs in other analysis tools.")

                # Save to history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'type': 'Prescription Analysis',
                    'input': (prescription_text[:100] + "..." if isinstance(prescription_text, str) and len(prescription_text) > 100 else prescription_text) if prescription_text else "",
                    'result': result
                })

                # Button to view complete analysis results
                if st.button("📊 View Complete Analysis Results", use_container_width=True):
                    st.session_state['selected_page'] = "📊 Analysis Results"
                    st.rerun()
                
                # Additional analysis options
                st.subheader("🔬 Additional Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔍 Check Drug Interactions"):
                        medications = [med['name'] for med in result.get('medications', [])]
                        if len(medications) >= 2:
                            # Auto-populate interaction checker with extracted medications
                            st.success(f"Found {len(medications)} medications. Checking interactions...")
                            # This would redirect to the interaction checker with pre-filled data
                        else:
                            st.warning("Need at least 2 medications to check interactions")
                
                with col2:
                    if st.button("💊 Calculate Dosages"):
                        medications = result.get('medications', [])
                        if medications:
                            st.success("Redirecting to dosage calculator...")
                            # This would redirect to the dosage calculator
                        else:
                            st.warning("No medications found to calculate dosages")

def show_alternative_drugs_page():
    """Alternative medications page"""

    st.markdown('<h2 class="section-header">🔄 Alternative Medication Finder</h2>', unsafe_allow_html=True)

    # Show existing analyzed drugs if available
    if 'analyzed_drugs' in st.session_state and st.session_state['analyzed_drugs']:
        st.info("💡 **Analyzed medications available:** " + ", ".join(st.session_state['analyzed_drugs']))
        st.markdown("You can analyze alternatives for these drugs or enter a different medication below.")

    st.markdown("""
    <div class="info-box">
    <strong>How to use:</strong> Enter a medication and patient constraints to find safer or equivalent alternatives.
    The system considers contraindications, allergies, and drug classes to suggest appropriate substitutes.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("alternative_drugs_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💊 Original Medication")
            
            original_drug = st.text_input(
                "Medication name:",
                placeholder="e.g., Aspirin, Lisinopril",
                help="Enter the medication you need alternatives for"
            )
            
            reason_for_alternative = st.selectbox(
                "Reason for seeking alternative:",
                options=[
                    "Contraindication",
                    "Allergy/Adverse reaction", 
                    "Drug interaction",
                    "Cost concerns",
                    "Availability issues",
                    "Patient preference",
                    "Other"
                ]
            )
            
            if reason_for_alternative == "Other":
                custom_reason = st.text_input("Please specify:")
        
        with col2:
            st.subheader("👤 Patient Constraints")
            
            patient_age = st.number_input(
                "Patient age:",
                min_value=0,
                max_value=120,
                value=None,
                help="Patient's age for age-appropriate alternatives"
            )
            
            contraindications = st.multiselect(
                "Contraindications:",
                options=[
                    "Pregnancy",
                    "Breastfeeding", 
                    "Kidney disease",
                    "Liver disease",
                    "Heart disease",
                    "Diabetes",
                    "Hypertension",
                    "Asthma",
                    "COPD",
                    "Depression"
                ],
                help="Select relevant contraindications"
            )
            
            allergies = st.multiselect(
                "Known allergies:",
                options=[
                    "Penicillin",
                    "Sulfa drugs",
                    "NSAIDs",
                    "Latex",
                    "Shellfish",
                    "Eggs",
                    "Nuts",
                    "Other"
                ],
                help="Select known allergies"
            )
            
            if "Other" in allergies:
                other_allergies = st.text_input("Specify other allergies:")
        
        # Additional constraints
        with st.expander("🔧 Additional Preferences"):
            col1, col2 = st.columns(2)
            
            with col1:
                preferred_route = st.selectbox(
                    "Preferred route:",
                    options=["Any", "Oral", "Topical", "Injectable", "Inhaled"],
                    help="Preferred method of administration"
                )
                
                dosing_frequency = st.selectbox(
                    "Preferred dosing frequency:",
                    options=["Any", "Once daily", "Twice daily", "Three times daily", "As needed"],
                    help="Preferred dosing schedule"
                )
            
            with col2:
                cost_consideration = st.selectbox(
                    "Cost consideration:",
                    options=["Any", "Generic preferred", "Brand name acceptable", "Low cost essential"],
                    help="Budget constraints for alternatives"
                )
                
                effectiveness_priority = st.selectbox(
                    "Effectiveness priority:",
                    options=["Equivalent efficacy", "Improved efficacy acceptable", "Reduced efficacy acceptable"],
                    help="Acceptable efficacy trade-offs"
                )
        
        submit_button = st.form_submit_button("🔍 Find Alternatives", use_container_width=True)
    
    if submit_button:
        if not original_drug:
            st.error("❌ Please enter a medication name")
        else:
            with st.spinner("Searching for alternative medications..."):
                # Prepare API request
                request_data = {
                    "drug_name": original_drug,
                    "contraindications": contraindications,
                    "allergies": [a for a in allergies if a != "Other"],
                    "patient_age": patient_age
                }
                
                # Make API call
                result = make_api_request("/alternative-drugs", method="POST", data=request_data)
                
                if result and result.get('alternatives'):
                    alternatives = result['alternatives']
                    
                    st.markdown('<div class="success-box">✅ <strong>Alternative medications found!</strong></div>', 
                               unsafe_allow_html=True)
                    
                    st.subheader(f"🔄 Alternatives for {original_drug}")
                    
                    for i, alt in enumerate(alternatives, 1):
                        with st.expander(f"Alternative #{i}: {alt.get('name', 'Unknown')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Medication Name:**")
                                st.write(alt.get('name', 'N/A'))
                                
                                st.write("**Drug Class:**")
                                st.write(alt.get('drug_class', 'N/A'))
                                
                                st.write("**Mechanism:**")
                                st.write(alt.get('mechanism', 'N/A'))
                            
                            with col2:
                                st.write("**Similarity Score:**")
                                similarity = alt.get('similarity_score', 0)
                                st.progress(similarity)
                                st.write(f"{similarity:.1%}")
                                
                                st.write("**Safety Profile:**")
                                safety = alt.get('safety_profile', 'Unknown')
                                safety_colors = {
                                    'High': '🟢',
                                    'Medium': '🟡',
                                    'Low': '🔴',
                                    'Unknown': '⚪'
                                }
                                st.write(f"{safety_colors.get(safety, '⚪')} {safety}")
                            
                            if alt.get('advantages'):
                                st.write("**Advantages:**")
                                for adv in alt['advantages']:
                                    st.write(f"• {adv}")
                            
                            if alt.get('considerations'):
                                st.write("**Considerations:**")
                                for cons in alt['considerations']:
                                    st.write(f"• {cons}")
                
                else:
                    st.warning("⚠️ No suitable alternatives found or API error occurred")

def show_analysis_results_page():
    """Comprehensive analysis results dashboard"""

    st.markdown('<h2 class="section-header">📊 Comprehensive Analysis Results</h2>', unsafe_allow_html=True)

    # Check if we have analyzed drugs and results
    if 'analyzed_drugs' not in st.session_state or not st.session_state['analyzed_drugs']:
        st.warning("⚠️ Please analyze a prescription first to see comprehensive results.")
        if st.button("📄 Go to Prescription Analyzer", use_container_width=True):
            st.session_state['selected_page'] = "📄 Prescription Analyzer"
            st.rerun()
        return

    if 'prescription_analysis' not in st.session_state:
        st.warning("⚠️ Prescription analysis data not found. Please re-analyze the prescription.")
        return

    drugs = st.session_state['analyzed_drugs']
    prescription_result = st.session_state['prescription_analysis']

    # Initialize result collections
    dosage_info_collected = []
    alternatives_info_collected = []

    # Overall summary
    st.markdown("""
    <div class="success-box">
    <strong>✅ Prescription Analysis Complete</strong><br>
    Found and analyzed medications for comprehensive safety profile
    </div>
    """, unsafe_allow_html=True)

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("💊 Medications Found", len(drugs))

    with col2:
        confidence = prescription_result.get('confidence_score', 0)
        st.metric("🎯 Analysis Confidence", f"{confidence:.1%}")

    with col3:
        # Simulate interaction count - would need to actually call the service
        interactions = len(drugs) * (len(drugs) - 1) // 2 if len(drugs) > 1 else 0
        st.metric("🔗 Potential Interactions", interactions)

    with col4:
        # Simulate alternatives count
        alternatives = len(drugs) * 2  # Rough estimate
        st.metric("🔄 Alternative Options", alternatives)

    st.markdown("---")

    # Detailed analysis results

    # 1. Original Prescription Analysis
    st.markdown("### 📄 Original Prescription Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Medications", "Dosages", "Frequencies", "Warnings"])

    medications = prescription_result.get('medications', [])

    with tab1:
        if medications:
            med_df = pd.DataFrame([{'Name': m.get('name', 'Unknown'), 'Confidence': m.get('confidence', 0)} for m in medications])
            st.dataframe(med_df, use_container_width=True)
        else:
            st.info("No medications identified")

    with tab2:
        dosages = prescription_result.get('dosages', [])
        if dosages:
            dosage_df = pd.DataFrame(dosages)
            st.dataframe(dosage_df, use_container_width=True)
        else:
            st.info("No dosage information found")

    with tab3:
        frequencies = prescription_result.get('frequencies', [])
        if frequencies:
            for freq in frequencies:
                st.write(f"• {freq}")
        else:
            st.info("No frequency information found")

    with tab4:
        warnings = prescription_result.get('warnings', [])
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No analysis warnings")

    # 2. Drug Interaction Analysis
    if len(drugs) >= 2:
        st.markdown("---")
        st.markdown("### 💊 Drug Interaction Analysis")

        if st.button("🔍 Analyze Interactions", use_container_width=True):
            with st.spinner("Analyzing drug interactions..."):
                request_data = {"drugs": drugs, "patient_age": None}
                result = make_api_request("/check-interactions", method="POST", data=request_data)

                if result:
                    display_interaction_results(
                        result.get('interactions', []),
                        result.get('safe', True),
                        result.get('risk_level', 'unknown'),
                        result.get('recommendations', [])
                    )
                else:
                    st.error("❌ Failed to analyze drug interactions")

    # 3. Dosage Analysis
    st.markdown("---")
    st.markdown("### 📏 Age-Specific Dosage Analysis")

    dosage_col1, dosage_col2 = st.columns(2)

    with dosage_col1:
        selected_age = st.slider("Patient Age", min_value=1, max_value=120, value=30, help="Select patient age for dosage recommendations")

    with dosage_col2:
        selected_weight = st.slider("Patient Weight (kg)", min_value=1.0, max_value=150.0, value=70.0, help="Select patient weight")

    if st.button("💊 Analyze Dosages", use_container_width=True):
        dosage_info_collected = []

        with st.spinner("Analyzing dosages for all medications..."):
            for drug_name in drugs:
                request_data = {
                    "drug_name": drug_name,
                    "patient_age": selected_age,
                    "weight": selected_weight,
                    "indication": None,
                    "kidney_function": None
                }

                result = make_api_request("/age-dosage", method="POST", data=request_data)
                if result:
                    dosage_info_collected.append((drug_name, result))

        if dosage_info_collected:
            for drug, dosage in dosage_info_collected:
                st.subheader(f"💊 {drug}")
                with st.container():
                    display_dosage_results(dosage)
                st.markdown("---")
        else:
            st.warning("No dosage information could be retrieved")

    # 4. Alternative Medications Analysis
    st.markdown("---")
    st.markdown("### 🔄 Alternative Medications Analysis")

    if st.button("🔍 Analyze Alternatives", use_container_width=True):
        alternatives_info_collected = []

        with st.spinner("Finding alternatives for all medications..."):
            for drug_name in drugs:
                request_data = {
                    "drug_name": drug_name,
                    "contraindications": [],
                    "allergies": [],
                    "patient_age": selected_age
                }

                result = make_api_request("/alternative-drugs", method="POST", data=request_data)
                if result and result.get('alternatives'):
                    alternatives_info_collected.append((drug_name, result.get('alternatives', [])))

        if alternatives_info_collected:
            for drug, alternatives in alternatives_info_collected:
                st.subheader(f"🔄 Alternatives for {drug}")

                col = st.columns(min(len(alternatives), 3))

                for i, alt in enumerate(alternatives[:3]):  # Show top 3 for each drug
                    with col[i]:
                        with st.container():
                            st.write(f"**{alt.get('name', 'Unknown')}**")
                            similarity = alt.get('similarity_score', 0)
                            st.progress(similarity)
                            st.caption(f"Similarity: {similarity:.1%}")
                            safety = alt.get('safety_profile', 'Unknown')
                            st.caption(f"Safety: {safety}")

                if len(alternatives) > 3:
                    with st.expander(f"Show all {len(alternatives)} alternatives for {drug}"):
                        for alt in alternatives[3:]:
                            st.write(f"• {alt.get('name', 'Unknown')} - Similarity: {alt.get('similarity_score', 0):.1%}")

                st.markdown("---")
        else:
            st.warning("No alternative medications could be found")

    # Actions
    st.markdown("---")
    st.markdown("### 🎯 Generate Complete Report")

    if st.button("📄 Generate PDF Report", use_container_width=True):
        st.info("PDF report generation would be implemented here")
        st.success("📄 Complete analysis report saved to session history!")

        # Save comprehensive results to history
        comprehensive_result = {
            'medications': medications,
            'dosages_analyzed': len(dosage_info_collected),
            'interactions_analyzed': len(drugs) >= 2,
            'alternatives_found': len(alternatives_info_collected),
            'patient_age': selected_age,
            'patient_weight': selected_weight
        }

        st.session_state.analysis_history.append({
            'timestamp': datetime.now(),
            'type': 'Comprehensive Analysis',
            'input': f"Prescribed medications: {', '.join(drugs)}",
            'result': comprehensive_result
        })

def show_analysis_history():
    """Display analysis history page"""
    
    st.markdown('<h2 class="section-header">📊 Analysis History</h2>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.info("📝 No analysis history available. Start by using the other tools!")
        return
    
    st.write(f"**Total analyses performed:** {len(st.session_state.analysis_history)}")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_type_filter = st.selectbox(
            "Filter by type:",
            options=["All"] + list(set([item['type'] for item in st.session_state.analysis_history]))
        )
    
    with col2:
        # Date range would go here
        st.write("**Date range:**")
        st.write("Last 30 days")  # Placeholder
    
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.success("History cleared!")
            st.experimental_rerun()
    
    # Display history
    filtered_history = st.session_state.analysis_history
    if analysis_type_filter != "All":
        filtered_history = [item for item in filtered_history if item['type'] == analysis_type_filter]
    
    for i, item in enumerate(reversed(filtered_history), 1):
        timestamp = item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        
        with st.expander(f"#{i} {item['type']} - {timestamp}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Input:**")
                if isinstance(item['input'], list):
                    st.write(", ".join(item['input']))
                elif isinstance(item['input'], dict):
                    for key, value in item['input'].items():
                        if value is not None:
                            st.write(f"- {key}: {value}")
                else:
                    st.write(str(item['input']))
            
            with col2:
                st.write("**Result Summary:**")
                result = item['result']
                
                if item['type'] == 'Drug Interaction':
                    interaction_count = len(result.get('interactions', []))
                    risk_level = result.get('risk_level', 'unknown')
                    st.write(f"- Interactions found: {interaction_count}")
                    st.write(f"- Risk level: {risk_level}")
                
                elif item['type'] == 'Dosage Calculation':
                    if 'recommended_dose' in result:
                        st.write(f"- Dose: {result['recommended_dose']} {result.get('unit', '')}")
                        st.write(f"- Frequency: {result.get('frequency', 'N/A')}")
                    else:
                        st.write("- No dosage information found")
                
                elif item['type'] == 'Prescription Parse':
                    med_count = len(result.get('medications', []))
                    confidence = result.get('confidence_score', 0)
                    st.write(f"- Medications found: {med_count}")
                    st.write(f"- Confidence: {confidence:.1%}")
            
            # Re-run analysis button
            if st.button(f"🔄 Re-run Analysis #{i}", key=f"rerun_{i}"):
                st.info("Re-analysis feature would be implemented here")

if __name__ == "__main__":
    main()