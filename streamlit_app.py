import streamlit as st #type:ignore
import pandas as pd
from datetime import datetime
import main as logic
import io
import zipfile
from dateutil.relativedelta import relativedelta
from datetime import date


def calculate_emi(pv, start_date, end_date, interest_rate, inflation_rate, current_date):
    # Calculate years to start (from current date to start date)
    years_to_start = (start_date - current_date).days / 365.25
    if years_to_start < 0: years_to_start = 0
    
    # Future Value of Loan Amount at Start Date
    fv_principal = pv * ((1 + inflation_rate/100) ** years_to_start)
    
    # Tenure in months
    delta = relativedelta(end_date, start_date)
    tenure_months = delta.years * 12 + delta.months
    
    if tenure_months <= 0:
        return 0
    
    # Monthly Interest Rate
    r = interest_rate / 12 / 100
    
    if r == 0:
        return fv_principal / tenure_months
        
    # EMI Formula
    emi = fv_principal * r * ((1 + r) ** tenure_months) / (((1 + r) ** tenure_months) - 1)
    
    return round(emi)

def main():

    st.set_page_config(page_title="Financial Planning", layout="wide")

    st.title("🎯 Financial Planning")

    # Initialize session state for goals and effects
    if 'goals' not in st.session_state:
        st.session_state.goals = []
    if 'effects' not in st.session_state:
        st.session_state.effects = []
    if 'custom_glide_paths' not in st.session_state:
        st.session_state.custom_glide_paths = {}
    if 'standard_glide_paths' not in st.session_state:
        st.session_state.standard_glide_paths = logic.get_default_glide_paths()
    if 'sip_adjustments' not in st.session_state:
        st.session_state.sip_adjustments = []

    # Section 1: Basic Information
    st.header("📊 Basic Information")
    col1, col2 = st.columns(2)

    with col1:
        # current_date = st.date_input("Current Date", value=datetime(2025, 1, 1))
        current_date = st.date_input("Current Date", value=date.today())
        current_corpus = st.number_input("Current Corpus (₹)", value=10000000, step=100000)
        yearly_sip_step_up = st.number_input("Yearly SIP Step-up (%)", value=10.0, step=0.1)

    with col2:
        current_age = st.number_input("Current Age", value=30, step=1)
        current_sip = st.number_input("Current SIP (₹/month)", value=100000, step=1000)

    st.divider()

    # Section 1.5: Advanced Options
    st.header("⚙️ Advanced Options")
    
    with st.expander("Advanced SIP Adjustments", expanded=False):
        st.subheader("Custom Yearly Step-Up Date")
        st.caption("By default, step-up happens every 12 months from the current date. Here you can specify a specific date when yearly step-up should occur.")
        
        col1, col2 = st.columns(2)
        with col1:
            use_custom_stepup = st.checkbox("Use Custom Step-Up Date", value=False, key="use_custom_stepup")
        
        if use_custom_stepup:
            with col1:
                stepup_month = st.selectbox(
                    "Step-Up Month",
                    options=list(range(1, 13)),
                    format_func=lambda x: pd.Timestamp(2000, x, 1).strftime('%B'),
                    key="stepup_month"
                )
            with col2:
                stepup_day = st.number_input(
                    "Step-Up Day",
                    min_value=1,
                    max_value=31,
                    value=1,
                    step=1,
                    key="stepup_day"
                )
        else:
            stepup_month = None
            stepup_day = None
        
        st.divider()
        
        st.subheader("Period-Based SIP Percentage Adjustments")
        st.caption("Apply percentage multipliers to SIP amounts for specific periods. For example, 150% means 1.5x the calculated SIP, 70% means 0.7x the SIP.")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("➕ Add Period Adjustment", use_container_width=True):
                st.session_state.sip_adjustments.append({
                    'id': len(st.session_state.sip_adjustments),
                    'start_date': datetime(2030, 1, 1),
                    'end_date': datetime(2035, 1, 1),
                    'percentage': 100.0
                })
                st.rerun()
        
        if len(st.session_state.sip_adjustments) == 0:
            st.info("No period adjustments added. Click 'Add Period Adjustment' to create one.")
        else:
            for i, adj in enumerate(st.session_state.sip_adjustments):
                with st.expander(f"Adjustment {i+1}: {adj['percentage']}% from {pd.Timestamp(adj['start_date']).strftime('%b %Y')} to {pd.Timestamp(adj['end_date']).strftime('%b %Y')}", expanded=True):
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    with col1:
                        adj['start_date'] = st.date_input(
                            "Start Date",
                            value=adj['start_date'],
                            min_value=pd.Timestamp(1900, 1, 1),
                            max_value=pd.Timestamp(2200, 12, 31),
                            key=f"adj_start_{i}"
                        )
                    
                    with col2:
                        adj['end_date'] = st.date_input(
                            "End Date",
                            value=adj['end_date'],
                            min_value=pd.Timestamp(1900, 1, 1),
                            max_value=pd.Timestamp(2200, 12, 31),
                            key=f"adj_end_{i}"
                        )
                    
                    with col3:
                        adj['percentage'] = st.number_input(
                            "Percentage (%)",
                            value=float(adj['percentage']),
                            min_value=0.0,
                            max_value=1000.0,
                            step=5.0,
                            key=f"adj_pct_{i}",
                            help="100% = no change, 150% = 1.5x SIP, 70% = 0.7x SIP"
                        )
                    
                    with col4:
                        st.write("")  # Spacing
                        if st.button("🗑️ Remove", key=f"remove_adj_{i}", use_container_width=True):
                            st.session_state.sip_adjustments.pop(i)
                            st.rerun()

    st.divider()

    # Section 2: Custom Glide Paths
    st.header("🛠️ Custom Glide Paths")
    
    with st.expander("Create New Glide Path"):
        gp_name = st.text_input("Glide Path Name", key="new_gp_name")
        
        if 'builder_buckets' not in st.session_state:
            st.session_state.builder_buckets = []
            
        if st.button("➕ Add Goal Value Bucket"):
            st.session_state.builder_buckets.append({
                'id': len(st.session_state.builder_buckets),
                'percent': 10.0,
                'steps': []
            })
            st.rerun()

        buckets_to_remove = []
        for i, bucket in enumerate(st.session_state.builder_buckets):
            # Collapsed by default unless it's the last one (likely being edited)
            is_expanded = (i == len(st.session_state.builder_buckets) - 1)
            
            with st.expander(f"Bucket {i+1} ({bucket['percent']}%)", expanded=is_expanded):
                col1, col2 = st.columns([3, 1])
                with col1:
                    bucket['percent'] = st.number_input(f"Percentage of Goal Value (%)", value=bucket['percent'], key=f"bucket_pct_{i}", label_visibility="collapsed")
                with col2:
                    if st.button("🗑️ Bucket", key=f"del_bucket_{i}"):
                        buckets_to_remove.append(i)
                
                # Steps
                if st.button(f"➕ Add Step", key=f"add_step_{i}"):
                    bucket['steps'].append({'instrument': 'debt', 'years': 0})
                    st.rerun()
                    
                steps_to_remove = []
                if bucket['steps']:
                    st.caption("Steps (Ordered by Years before maturity)")
                    for j, step in enumerate(bucket['steps']):
                        c1, c2, c3 = st.columns([2, 2, 1])
                        with c1:
                            step['instrument'] = st.selectbox(f"Instrument", options=['hybrid', 'debt'], index=0 if step['instrument']=='hybrid' else 1, key=f"step_inst_{i}_{j}", label_visibility="collapsed")
                        with c2:
                            step['years'] = st.number_input(f"Years Before Maturity", value=step['years'], min_value=0, step=1, key=f"step_years_{i}_{j}", label_visibility="collapsed")
                        with c3:
                            if st.button("🗑️", key=f"del_step_{i}_{j}"):
                                steps_to_remove.append(j)
                
                for index in sorted(steps_to_remove, reverse=True):
                    bucket['steps'].pop(index)

        for index in sorted(buckets_to_remove, reverse=True):
            st.session_state.builder_buckets.pop(index)
            st.rerun()

        if st.button("💾 Save Custom Glide Path", type="primary"):
            total_percent = sum(b['percent'] for b in st.session_state.builder_buckets)

            if not gp_name:
                st.error("Please provide a name.")
            elif not st.session_state.builder_buckets:
                st.error("Please add at least one bucket.")
            elif abs(total_percent - 100.0) > 0.01:
                st.error(f"Total percentage must be 100%. Current total: {total_percent}%")
            else:
                # Logic to convert builder to DataFrame
                rows = []
                row_id_counter = 1
                
                valid = True
                for bucket in st.session_state.builder_buckets:
                    if not bucket['steps']:
                        st.error("Every bucket must have at least one step.")
                        valid = False
                        break
                    
                    # Sort steps by years ascending (closest to goal first? No, we want logical flow)
                    # We decided: Steps are "Start Year".
                    # Step 1: Debt, 2 years. Means 2 years before end.
                    # Step 2: Hybrid, 10 years. Means 10 years before end.
                    # Flow: Core -> Hybrid (at 10) -> Debt (at 2) -> Goal (at 0).
                    # So we process steps in ASCENDING order of years (smallest first).
                    # Smallest year (e.g. 2) connects to Goal (0).
                    # Next smallest (e.g. 10) connects to Previous (2).
                    
                    sorted_steps = sorted(bucket['steps'], key=lambda x: x['years'])
                    
                    # Ensure strictly increasing if desired, or at least distinct.
                    # 0 years is valid? 0 years before maturity is maturity.
                    # If step year is 0, it means it IS the goal? No.
                    # It means it starts at 0 years before. But goal is at 0.
                    # So duration is 0? That's useless.
                    # Validation: Years should be > 0.
                    
                    previous_node_id = None
                    last_start_year = 0
                    
                    # Create Goal Row first?
                    # No, goal row is the sink.
                    # In existing logic, 'inflow_from' points to source.
                    # So we need Source ID.
                    
                    # Let's generate IDs for steps first.
                    # But we need to link them.
                    
                    # Let's build the chain.
                    # Chain: Core -> Step N -> ... -> Step 1 -> Goal
                    
                    # Step N is the one with LARGEST start year.
                    # Step 1 is the one with SMALLEST start year.
                    
                    # Let's iterate reversed sorted steps (Largest to Smallest).
                    reverse_sorted = sorted(bucket['steps'], key=lambda x: x['years'], reverse=True)
                    
                    current_source = 'core corpus'
                    
                    for k, step in enumerate(reverse_sorted):
                        # Create row for this step
                        step_row = {
                            'id': row_id_counter,
                            'place': step['instrument'],
                            'years from inflow till end': step['years'],
                            'years from outflow till end': 0 if k == len(reverse_sorted) - 1 else reverse_sorted[k+1]['years'], # Outflow to next step (which has smaller year) or 0 (Goal)
                            'inflow_from': current_source,
                            # outflow_to is ID of next step... wait.
                            # The dataframe has 'outflow_to' but `process_chain` uses `inflow_from`.
                            # `main.py` logic traces BACKWARDS from goal.
                            # So Goal.inflow_from = Step1_ID.
                            # Step1.inflow_from = Step2_ID.
                            # StepN.inflow_from = 'core corpus'.
                            
                            # So I need the IDs of these steps.
                        }
                        # Wait, I can't know inflow_from if I haven't created the source yet?
                        # No, if source is core corpus, I know it.
                        # If source is previous step, I know its ID if I created it.
                        
                        # So iterate from Furthest (Core source) to Closest (Goal source).
                        # Correct.
                        pass # Placeholder logic check
                    
                    # Let's do it properly.
                    # Sort steps Descending (High year to Low year).
                    # Example: Hybrid (10), Debt (2).
                    
                    # Step A: Hybrid (10). Source: Core.
                    # Step B: Debt (2). Source: Step A.
                    # Goal. Source: Step B.
                    
                    chain_ids = []
                    
                    sorted_desc = sorted(bucket['steps'], key=lambda x: x['years'], reverse=True)
                    
                    last_id = 'core corpus' # Start source
                    
                    for step in sorted_desc:
                        current_id = row_id_counter
                        row_id_counter += 1
                        
                        # Calculate outflow year (next step's inflow year, or 0 if last)
                        # Actually we don't strictly need 'outflow_to' or 'years from outflow' for logic?
                        # main.py logic:
                        # process_chain navigates via `inflow_from`.
                        # It calculates `years = (outflow_date - inflow_date)`.
                        # `outflow_date` is `current_row['inflow_date']` (i.e. the node that pulled from this source).
                        # So `years from outflow till end` in the excel seems to be for documentation or validation, `main.py` line 29 calculates `outflow_date` but line 86 uses `current_row['inflow_date']` as the outflow date of the source.
                        # Wait, line 86: `outflow_date = current_row['inflow_date']`.
                        # `current_row` is the receiver. `source_row` is the provider.
                        # Provider's outflow date IS receiver's inflow date.
                        # So we just need to set up `inflow_from` links correctly.
                        
                        # However, for `years from outflow till end`, let's populate it for consistency.
                        # If this is Step A (High Year), its outflow is Step B (Low Year).
                        # If this is Step Last (Lowest Year), its outflow is Goal (0 Year).
                        
                        # But wait, `main.py` line 29: df['outflow_date'] = ... years from outflow till end.
                        # Is `outflow_date` used?
                        # Line 133: `if pd.notna(row['outflow_date']):` - used for calculation of growth/tax in INTERMEDIATE steps (not goal).
                        # Yes! `process_chain` sets `inflow_amount` on source.
                        # Later (line 119+), `df.iterrows()` calculates growth on non-goal rows using `outflow_date`.
                        # So `years from outflow till end` IS CRITICAL.
                        
                        # So:
                        # Step A (10): Outflow year = Step B's inflow year (2).
                        # Step B (2): Outflow year = Goal's inflow year (0).
                        
                        rows.append({
                            'id': current_id,
                            'place': step['instrument'],
                            'years from inflow till end': step['years'],
                            'years from outflow till end': 0 if step == sorted_desc[-1] else 0, # To be filled?
                            # 'years from outflow till end' needs to be the inflow year of the NEXT step in the chain (closer to goal).
                            'inflow_from': last_id,
                            'outflow_to': 0, # Placeholder
                            '% of goal value': bucket['percent'] # Optional, only needed for goal row
                        })
                        chain_ids.append(current_id)
                        last_id = current_id
                    
                    # Post-process to fix 'years from outflow' and 'outflow_to'
                    for k in range(len(chain_ids)):
                        curr_idx = len(rows) - len(chain_ids) + k
                        # If not last step
                        if k < len(chain_ids) - 1:
                            next_id = chain_ids[k+1]
                            # Next step in list is the one Closest to goal (next in iteration was closer).
                            # Wait, sorted_desc is Furthest -> Closest.
                            # So chain_ids[0] is Furthest.
                            # chain_ids[1] is Closer.
                            # Step 0 flows to Step 1.
                            rows[curr_idx]['outflow_to'] = chain_ids[k+1]
                            rows[curr_idx]['years from outflow till end'] = rows[curr_idx+1]['years from inflow till end']
                        else:
                            # Last step flows to Goal
                            # Goal inflow year is 0.
                            rows[curr_idx]['outflow_to'] = 'Goal' # placeholder logic
                            rows[curr_idx]['years from outflow till end'] = 0
                            
                    # Create Goal Row
                    goal_id = row_id_counter
                    row_id_counter += 1
                    rows.append({
                        'id': goal_id,
                        'place': 'goal',
                        'years from inflow till end': 0,
                        'years from outflow till end': pd.NA,
                        'inflow_from': last_id, # The last step ID
                        'outflow_to': pd.NA,
                        '% of goal value': bucket['percent']
                    })
                
                if valid:
                    # Create DataFrame
                    df_custom = pd.DataFrame(rows)
                    # df_custom.to_excel('Custom Glide Paths.xlsx', sheet_name=gp_name, index=False)
                    st.session_state.custom_glide_paths[gp_name] = df_custom
                    st.success(f"Created Glide Path: {gp_name}")
                    st.session_state.builder_buckets = [] # Reset
                    st.rerun()

    # Section 2.5: Financial Goals
    st.header("🎯 Financial Goals")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Add Goal", use_container_width=True):
            st.session_state.goals.append({
                'id': len(st.session_state.goals),
                'name': '',
                'type': 'Non-Negotiable',
                'maturity_date': datetime(2040, 1, 1),
                'downpayment_present_value': 5000000,
                'rate_for_future_value': 6.0
            })
            st.rerun()

    if len(st.session_state.goals) == 0:
        st.info("No goals added yet. Click 'Add Goal' to get started.")
    else:
        for i, goal in enumerate(st.session_state.goals):
            with st.expander(f"Goal {i+1}: {goal['name'] if goal['name'] else 'Unnamed'}", expanded=True):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    goal['name'] = st.text_input(
                        "Goal Name", 
                        value=goal['name'], 
                        key=f"goal_name_{i}",
                        placeholder="e.g., Home, Child Education"
                    )
                    goal['downpayment_present_value'] = st.number_input(
                        "Present Value (₹)", 
                        value=goal['downpayment_present_value'], 
                        step=100000,
                        key=f"goal_pv_{i}"
                    )
                
                with col2:
                    gp_options = list(st.session_state.standard_glide_paths.keys()) + list(st.session_state.custom_glide_paths.keys())
                    current_type = goal['type'] if goal['type'] in gp_options else gp_options[0]
                    
                    goal['type'] = st.selectbox(
                        "Goal Type", 
                        options=gp_options,
                        index=gp_options.index(current_type),
                        key=f"goal_type_{i}"
                    )
                    goal['rate_for_future_value'] = st.number_input(
                        "Inflation Rate (%)", 
                        value=goal['rate_for_future_value'], 
                        step=0.1,
                        key=f"goal_rate_{i}"
                    )
                
                with col3:
                    goal['maturity_date'] = st.date_input(
                        "Maturity Date", 
                        value=goal['maturity_date'],
                        min_value=pd.Timestamp(1900, 1, 1),
                        max_value=pd.Timestamp(2200, 12, 31),
                        key=f"goal_date_{i}"
                    )
                    if st.button("🗑️ Remove", key=f"remove_goal_{i}", use_container_width=True):
                        st.session_state.goals.pop(i)
                        st.rerun()

    st.divider()

    # Section 3: Effects on Cashflows
    st.header("💰 Effects on Cashflows")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Add Cashflow Effect", use_container_width=True):
            st.session_state.effects.append({
                'id': len(st.session_state.effects),
                'type': 'Manual', # Default
                'start_date': datetime(2030, 1, 1),
                'end_date': datetime(2050, 1, 1),
                'monthly_amount': -30000,
                'pv': 5000000,
                'interest_rate': 8.5,
                'inflation_rate': 6.0
            })
            st.rerun()

    if len(st.session_state.effects) == 0:
        st.info("No cashflow effects added yet. Click 'Add Cashflow Effect' to get started.")
    else:
        for i, effect in enumerate(st.session_state.effects):
            with st.expander(f"Cashflow Effect {i+1}", expanded=True):
                # Ensure keys exist for backward compatibility
                effect.setdefault('type', 'Manual')
                
                type_col, _ = st.columns([1, 3])
                with type_col:
                    effect['type'] = st.selectbox("Type", ["Manual", "Loan EMI"], key=f"effect_type_{i}")
                
                if effect['type'] == "Manual":
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    with col1:
                        effect['start_date'] = st.date_input(
                            "Start Date", 
                            value=effect['start_date'],
                            min_value=pd.Timestamp(1900, 1, 1),
                            max_value=pd.Timestamp(2200, 12, 31),
                            key=f"effect_start_{i}"
                        )
                    
                    with col2:
                        effect['end_date'] = st.date_input(
                            "End Date", 
                            value=effect['end_date'],
                            min_value=pd.Timestamp(1900, 1, 1),
                            max_value=pd.Timestamp(2200, 12, 31),
                            key=f"effect_end_{i}"
                        )
                    
                    with col3:
                        effect['monthly_amount'] = st.number_input(
                            "Monthly Amount (₹)", 
                            value=int(effect['monthly_amount']),
                            step=1000,
                            key=f"effect_amount_{i}",
                            help="Use negative values for outflow"
                        )
                    
                    with col4:
                        st.write("")  # Spacing
                        if st.button("🗑️ Remove", key=f"remove_effect_{i}", use_container_width=True):
                            st.session_state.effects.pop(i)
                            st.rerun()
                            
                else: # Loan EMI
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        effect.setdefault('pv', 5000000)
                        effect['pv'] = st.number_input("Loan Amount (PV)", value=effect['pv'], step=100000, key=f"eff_pv_{i}")
                    with c2:
                        effect.setdefault('interest_rate', 8.5)
                        effect['interest_rate'] = st.number_input("Loan Interest (%)", value=effect['interest_rate'], step=0.1, key=f"eff_int_{i}")
                    with c3:
                        effect.setdefault('inflation_rate', 6.0)
                        effect['inflation_rate'] = st.number_input("Inflation Rate (%)", value=effect['inflation_rate'], step=0.1, key=f"eff_inf_{i}")
                        
                    c4, c5, c6 = st.columns([2, 2, 2])
                    with c4:
                        effect['start_date'] = st.date_input("Loan Start Date", value=effect['start_date'], key=f"eff_st_{i}")
                    with c5:
                        effect['end_date'] = st.date_input("Loan End Date", value=effect['end_date'], key=f"eff_end_{i}")
                        
                    # Calculate EMI
                    current_date_ts = pd.Timestamp(st.session_state.get('current_date_input', datetime(2025, 12, 23)))
                    # Need to access current_date from input section. It's properly in 'current_date' variable from line 25 if scope allows?
                    # Streamlit reruns script top to bottom. `current_date` is defined in main() scope. We are inside main(). So yes.
                    
                    calculated_emi = calculate_emi(
                        effect['pv'], 
                        pd.Timestamp(effect['start_date']), 
                        pd.Timestamp(effect['end_date']), 
                        effect['interest_rate'], 
                        effect['inflation_rate'], 
                        pd.Timestamp(current_date)
                    )
                    
                    # Store as negative monthly amount
                    effect['monthly_amount'] = -calculated_emi
                    
                    with c6:
                        st.metric("Calculated EMI", logic.format_inr(calculated_emi))
                        
                    if st.button("🗑️ Remove", key=f"remove_effect_emi_{i}", use_container_width=True):
                        st.session_state.effects.pop(i)
                        st.rerun()

    st.divider()

    # Section 4: Instrument Parameters
    st.header("⚙️ Instrument Parameters")

    with st.expander("Configure Instrument Returns and Taxes", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hybrid")
            hybrid_return = st.number_input("Return (%)", value=10.0, step=0.1, key="hybrid_return")
            hybrid_tax = st.number_input("Tax (%)", value=12.5, step=0.1, key="hybrid_tax")
            
            st.subheader("Debt")
            debt_return = st.number_input("Return (%)", value=6.0, step=0.1, key="debt_return")
            debt_tax = st.number_input("Tax (%)", value=30.0, step=0.1, key="debt_tax")

        with col2:
            st.subheader("Core Corpus")
            core_return = st.number_input("Return (%)", value=15.0, step=0.1, key="core_return")
            core_tax = st.number_input("Tax (%)", value=12.5, step=0.1, key="core_tax")
    
    # Goal return and tax are always 0 (not shown to user)
    goal_return = 0.0
    goal_tax = 0.0

    st.divider()

    # Generate Output Button
    if st.button("🚀 Show Simulation Results", type="primary", use_container_width=True):
        
        # Create input_variables dictionary
        input_variables = {
            'current_date': pd.Timestamp(current_date),
            'current_age': int(current_age),
            'current_corpus': int(current_corpus),
            'current_sip': int(current_sip),
            'yearly_sip_step_up_%': float(yearly_sip_step_up),
            'stepup_date_month': stepup_month,
            'stepup_date_day': stepup_day,
            'sip_adjustments': [
                {
                    'start_date': pd.Timestamp(adj['start_date']),
                    'end_date': pd.Timestamp(adj['end_date']),
                    'percentage': float(adj['percentage'])
                }
                for adj in st.session_state.sip_adjustments
            ],
            'goals': [
                {
                    'name': goal['name'],
                    'type': goal['type'],
                    'maturity_date': pd.Timestamp(goal['maturity_date']),
                    'downpayment_present_value': int(goal['downpayment_present_value']),
                    'rate_for_future_value%': float(goal['rate_for_future_value'])
                }
                for goal in st.session_state.goals
            ],
            'effects_on_cashflows': [
                {
                    'start_date': pd.Timestamp(effect['start_date']),
                    'end_date': pd.Timestamp(effect['end_date']),
                    'monthly_amount': int(effect['monthly_amount'])
                }
                for effect in st.session_state.effects
            ]
        }
        
        # Create instrument_params dictionary (convert percentages to decimals)
        instrument_params = {
            'hybrid': {'return': hybrid_return / 100, 'tax': hybrid_tax / 100},
            'debt': {'return': debt_return / 100, 'tax': debt_tax / 100},
            'goal': {'return': goal_return / 100, 'tax': goal_tax / 100},
            'core_corpus': {'return': core_return / 100, 'tax': core_tax / 100}
        }
        
        # Combine glide paths
        all_glide_paths = st.session_state.standard_glide_paths.copy()
        all_glide_paths.update(st.session_state.custom_glide_paths)
        
        # Run simulation
        result = logic.run_simulation(input_variables, instrument_params, glide_paths=all_glide_paths)
        
        if result:
            if result['status'] == 'error':
                st.error(result['message'])
                if 'sip_df' in result['data']:
                    st.subheader('Please look at the adjustments in cashflows')
                    st.dataframe(result['data']['sip_df'])
            
            elif result['status'] in ['success', 'failure']:
                data = result['data']
                success_metrics = data['success_metrics']
                daily_corpus_value_df = data['daily_corpus_value_df']
                final_trans_df = data['final_trans_df']
                goal_dfs = data['goal_dfs']
                last_goal_date = data['last_goal_date']
                
                if result['status'] == 'success':
                    st.success('All goals met Successfully')
                    
                    st.header(
                        f"Corpus on {last_goal_date.strftime('%d-%b-%Y')} "
                        f"will be {logic.format_inr(daily_corpus_value_df['current_value'].iloc[-1])}"
                    )
                    
                    st.subheader("Daily Corpus Value")
                    st.line_chart(daily_corpus_value_df, x='Date', y='current_value')

                    st.divider()

                    # Create zip file in memory
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zf:
                        # Helper to add df to zip
                        def add_df_to_zip(df, name):
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            zf.writestr(f"{name}.csv", csv_data)

                        add_df_to_zip(daily_corpus_value_df, "daily_corpus_value")
                        add_df_to_zip(final_trans_df, "final_trans_df")
                        add_df_to_zip(data.get('consolidated_trans_df', pd.DataFrame()), "consolidated_trans_df")
                        add_df_to_zip(data.get('nav_df', pd.DataFrame()), "nav_df")
                        add_df_to_zip(data.get('sip_df', pd.DataFrame()), "sip_df")
                        add_df_to_zip(data.get('sip_trans_df', pd.DataFrame()), "sip_trans_df")
                        add_df_to_zip(data.get('withdrawls_df', pd.DataFrame()), "withdrawls_df")
                        
                        for goal_name, goal_df in goal_dfs.items():
                             add_df_to_zip(goal_df, f"goal_{goal_name}")
                    
                    st.download_button(
                        label="Download All Data (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="simulation_results.zip",
                        mime="application/zip"
                    )

                else:
                    st.error('Goals could not be met')
                    st.header(f"Cashflows started going negative on {success_metrics['simulation_broke_date'].strftime('%d-%b-%Y')}")


                # st.subheader('Cashflows in core corpus:')
                # st.dataframe(final_trans_df)

                # st.subheader('Goal wise cashflows:')
                # for goal_name, goal_df in goal_dfs.items():
                #     st.write(goal_name)
                #     st.dataframe(goal_df)

if __name__ == "__main__":
    main()
