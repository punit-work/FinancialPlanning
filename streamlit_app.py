import streamlit as st #type:ignore
import pandas as pd
from datetime import datetime
import main_v2 as logic
# import io
# import zipfile
from dateutil.relativedelta import relativedelta
from datetime import date


def parse_date(date_str, default=None):
    """Parse date string in dd/mm/yyyy format. Returns default if parsing fails."""
    if not date_str:
        return default
    try:
        return datetime.strptime(date_str.strip(), "%d/%m/%Y").date()
    except ValueError:
        return default


def format_date_for_input(d):
    """Format a date object to dd/mm/yyyy string for display in text input."""
    if isinstance(d, (date, datetime, pd.Timestamp)):
        if isinstance(d, pd.Timestamp):
            d = d.date()
        elif isinstance(d, datetime):
            d = d.date()
        return d.strftime("%d/%m/%Y")
    return ""


def date_text_input(label, value, key, help_text=None):
    """
    Replacement for st.date_input that uses text input.
    Returns a date object or None if invalid.
    """
    default_str = format_date_for_input(value)
    placeholder = "dd/mm/yyyy"
    
    if help_text:
        full_help = f"{help_text} (Format: dd/mm/yyyy)"
    else:
        full_help = "Format: dd/mm/yyyy"
    
    text_val = st.text_input(label, value=default_str, key=key, help=full_help, placeholder=placeholder)
    
    parsed = parse_date(text_val, value)
    return parsed


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

    # Initialize session state for goals, effects, and adjustments
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
    if 'expense_streams' not in st.session_state:
        st.session_state.expense_streams = [{
            'name': 'Living Expenses',
            'amount': 50000,
            'frequency': 1,
            'inflation': 6.0,
            'post_retirement_change': -15.0,
            'adjustments': []
        }]

    # Section 1: Basic Information
    st.header("📊 Basic Information")
    col1, col2 = st.columns(2)

    with col1:
        current_date = date_text_input("Current Date", value=date.today(), key="current_date")
        current_corpus = st.number_input("Current Corpus (₹)", value=10000000, step=100000)
        yearly_sip_step_up = st.number_input("Yearly SIP Step-up (%)", value=10.0, step=0.1)

    with col2:
        current_age = st.number_input("Current Age", value=30, step=1)
        target_lifetime = st.number_input("Target Lifetime (Years)", value=100, min_value=int(current_age) + 1, step=1)
        current_sip = st.number_input("Current SIP (₹/month)", value=100000, step=1000)

    death_date = pd.Timestamp(current_date) + pd.DateOffset(years=int(target_lifetime - current_age))

    st.divider()
    
    # New Section: Expense Streams
    st.header("💸 Living Expenses & Streams")
    st.caption("Define multiple streams of expenses (e.g., Household, Travel, Medical) with their own inflation and adjustments.")
    
    if st.button("➕ Add Expense Stream"):
        st.session_state.expense_streams.append({
            'name': f"Stream {len(st.session_state.expense_streams)+1}",
            'amount': 20000,
            'frequency': 1, # Monthly
            'inflation': 6.0,
            'post_retirement_change': 0.0,
            'adjustments': []
        })
        st.rerun()

    if len(st.session_state.expense_streams) == 0:
        st.warning("No expense streams defined. Please add at least one.")
    
    streams_to_remove = []
    
    def freq_label(f):
        if f == 1: return "Monthly"
        if f == 3: return "Quarterly"
        if f == 6: return "Semi-Annually"
        if f == 12: return "Yearly"
        return f"Every {f} Months"
    
    for i, stream in enumerate(st.session_state.expense_streams):
        with st.expander(f"{stream['name']} ({logic.format_inr(stream['amount'])} - {freq_label(stream['frequency'])})", expanded=False):
            
            # Row 1: Basic Config
            c1, c2, c3 = st.columns(3)
            with c1:
                stream['name'] = st.text_input(f"Name", value=stream['name'], key=f"s_name_{i}")
                stream['amount'] = st.number_input(f"Amount (₹)", value=int(stream['amount']), step=1000, key=f"s_amt_{i}")
            
            with c2:
                stream['frequency'] = st.number_input(f"Frequency (Months)", value=int(stream['frequency']), min_value=1, step=1, key=f"s_freq_{i}", help="1=Monthly, 3=Quarterly, 12=Yearly")
                stream['inflation'] = st.number_input(f"Inflation (%)", value=float(stream['inflation']), step=0.1, key=f"s_inf_{i}")
                
            with c3:
                stream['post_retirement_change'] = st.number_input(f"Step Up after Ret. (%)", value=float(stream['post_retirement_change']), step=1.0, help="Percentage change in expense after retirement (e.g. -15 for reduction, +20 for travel)", key=f"s_prc_{i}")
                if st.button("🗑️ Delete Stream", key=f"del_stream_{i}"):
                    streams_to_remove.append(i)
            
            # Row 2: Adjustments
            st.markdown("---")
            st.caption(f"Adjustments for **{stream['name']}**")
            
            if st.button(f"➕ Add Adjustment", key=f"add_adj_{i}"):
                stream['adjustments'].append({
                    'start_date': date(2035, 1, 1),
                    'end_date': date(2040, 1, 1),
                    'type': 'Multiplier', # Multiplier or Fixed
                    'value': 100.0
                })
                st.rerun()
                
            adj_to_remove = []
            if stream['adjustments']:
                for j, adj in enumerate(stream['adjustments']):
                    ac1, ac2, ac3, ac4, ac5 = st.columns([2, 2, 2, 2, 1])
                    adj.setdefault('type', 'Multiplier')
                    
                    with ac1: 
                        adj['start_date'] = date_text_input("Start", value=adj['start_date'], key=f"s_{i}_adj_s_{j}")
                    with ac2: 
                        adj['end_date'] = date_text_input("End", value=adj['end_date'], key=f"s_{i}_adj_e_{j}")
                    with ac3: 
                        adj['type'] = st.selectbox("Type", ["Multiplier", "Fixed Amount"], index=0 if adj['type']=='Multiplier' else 1, key=f"s_{i}_adj_t_{j}", label_visibility="collapsed")
                    with ac4: 
                        if adj['type'] == 'Multiplier':
                            # Show Percentage Input
                            adj['value'] = st.number_input("%", value=float(adj['value']), step=5.0, key=f"s_{i}_adj_v_{j}", label_visibility="collapsed", help="% Multiplier")
                        else:
                            # Show Amount Input
                            adj['value'] = st.number_input("₹", value=float(adj['value']), step=1000.0, key=f"s_{i}_adj_v_{j}", label_visibility="collapsed", help="Fixed Adjustment Amount (+/-)")
                            
                    with ac5: 
                        if st.button("🗑️", key=f"s_{i}_del_adj_{j}"):
                            adj_to_remove.append(j)
                            
                for idx in sorted(adj_to_remove, reverse=True):
                    stream['adjustments'].pop(idx)
                    st.rerun()
                    
    for idx in sorted(streams_to_remove, reverse=True):
        st.session_state.expense_streams.pop(idx)
        st.rerun()

    st.divider()

    # Section 1.5: Advanced Options
    st.header("⚙️ Advanced Options")
    
    with st.expander("Advanced SIP Settings", expanded=False):
        
        tab1, tab2 = st.tabs(["SIP Step-Up", "SIP Adjustments"])
        
        with tab1:
            st.subheader("Custom Yearly Step-Up Date")
            st.caption("By default, step-up happens every 12 months from the current date. Here you can specify a specific date when yearly step-up should occur.")
            
            t1_col1, t1_col2 = st.columns(2)
            with t1_col1:
                use_custom_stepup = st.checkbox("Use Custom Step-Up Date", value=False, key="use_custom_stepup")
            
            if use_custom_stepup:
                with t1_col1:
                    stepup_month = st.selectbox(
                        "Step-Up Month",
                        options=list(range(1, 13)),
                        format_func=lambda x: pd.Timestamp(2000, x, 1).strftime('%B'),
                        key="stepup_month"
                    )
                with t1_col2:
                    stepup_day = st.number_input(
                        "Step-Up Day", min_value=1, max_value=31, value=1, step=1, key="stepup_day"
                    )
            else:
                stepup_month = None
                stepup_day = None
        
        with tab2:
            st.subheader("Period-Based SIP Percentage Adjustments")
            st.caption("Apply percentage multipliers to SIP amounts for specific periods.")
            
            if st.button("➕ Add SIP Adjustment", use_container_width=True):
                st.session_state.sip_adjustments.append({
                    'start_date': datetime(2030, 1, 1),
                    'end_date': datetime(2035, 1, 1),
                    'percentage': 100.0
                })
                st.rerun()
            
            if len(st.session_state.sip_adjustments) == 0:
                st.info("No SIP adjustments added.")
            else:
                for i, adj in enumerate(st.session_state.sip_adjustments):
                    with st.container():
                        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                        with c1:
                            adj['start_date'] = date_text_input("Start", value=adj['start_date'], key=f"s_adj_start_{i}")
                        with c2:
                            adj['end_date'] = date_text_input("End", value=adj['end_date'], key=f"s_adj_end_{i}")
                        with c3:
                            adj['percentage'] = st.number_input(f"% Multiplier", value=float(adj['percentage']), step=5.0, key=f"s_adj_pct_{i}")
                        if c4.button("🗑️", key=f"del_s_adj_{i}"):
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
                rows = []
                row_id_counter = 1
                valid = True
                for bucket in st.session_state.builder_buckets:
                    if not bucket['steps']:
                        st.error("Every bucket must have at least one step.")
                        valid = False
                        break
                    
                    sorted_desc = sorted(bucket['steps'], key=lambda x: x['years'], reverse=True)
                    chain_ids = []
                    last_id = 'core corpus' 
                    
                    for step in sorted_desc:
                        current_id = row_id_counter
                        row_id_counter += 1
                        rows.append({
                            'id': current_id,
                            'place': step['instrument'],
                            'years from inflow till end': step['years'],
                            'years from outflow till end': 0, # Placeholder
                            'inflow_from': last_id,
                            'outflow_to': 0, 
                            '% of goal value': bucket['percent']
                        })
                        chain_ids.append(current_id)
                        last_id = current_id
                    
                    for k in range(len(chain_ids)):
                        curr_idx = len(rows) - len(chain_ids) + k
                        if k < len(chain_ids) - 1:
                            rows[curr_idx]['outflow_to'] = chain_ids[k+1]
                            rows[curr_idx]['years from outflow till end'] = rows[curr_idx+1]['years from inflow till end']
                        else:
                            rows[curr_idx]['outflow_to'] = 'Goal'
                            rows[curr_idx]['years from outflow till end'] = 0
                            
                    goal_id = row_id_counter
                    row_id_counter += 1
                    rows.append({
                        'id': goal_id,
                        'place': 'goal',
                        'years from inflow till end': 0,
                        'years from outflow till end': pd.NA,
                        'inflow_from': last_id,
                        'outflow_to': pd.NA,
                        '% of goal value': bucket['percent']
                    })
                
                if valid:
                    df_custom = pd.DataFrame(rows)
                    st.session_state.custom_glide_paths[gp_name] = df_custom
                    st.success(f"Created Glide Path: {gp_name}")
                    st.session_state.builder_buckets = []
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
            # Calculate future value for display
            goal_fv = logic.future_value(
                goal['downpayment_present_value'], 
                goal['rate_for_future_value'] / 100, 
                pd.Timestamp(current_date), 
                pd.Timestamp(goal['maturity_date'])
            )
            maturity_str = pd.Timestamp(goal['maturity_date']).strftime('%b %Y')
            expander_label = f"Goal {i+1}: {goal['name'] if goal['name'] else 'Unnamed'} — FV: {logic.format_inr(goal_fv)} @ {maturity_str}"
            
            with st.expander(expander_label, expanded=True):
                if pd.Timestamp(goal['maturity_date']) >= death_date:
                    st.error(f"Goal matures at or after target lifetime ({death_date.strftime('%b %Y')}). Please adjust the maturity date.")

                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    goal['name'] = st.text_input(f"Goal Name {i}", value=goal['name'], key=f"goal_name_{i}", placeholder="e.g., Home")
                    goal['downpayment_present_value'] = st.number_input(f"Present Value (₹) {i}", value=goal['downpayment_present_value'], step=100000, key=f"goal_pv_{i}")
                
                with col2:
                    gp_options = list(st.session_state.standard_glide_paths.keys()) + list(st.session_state.custom_glide_paths.keys())
                    current_type = goal['type'] if goal['type'] in gp_options else gp_options[0]
                    goal['type'] = st.selectbox(f"Goal Type {i}", options=gp_options, index=gp_options.index(current_type), key=f"goal_type_{i}")
                    goal['rate_for_future_value'] = st.number_input(f"Inflation (%) {i}", value=goal['rate_for_future_value'], step=0.1, key=f"goal_rate_{i}")
                
                with col3:
                    goal['maturity_date'] = date_text_input(
                        f"Maturity {i}", 
                        value=goal['maturity_date'],
                        key=f"goal_date_{i}"
                    )
                    if st.button("🗑️ Remove", key=f"remove_goal_{i}", use_container_width=True):
                        st.session_state.goals.pop(i)
                        st.rerun()

    st.divider()

    # Section 3: Effects on Cashflows (SIP Capacity Effects)
    st.header("💰 Effects on SIP Capacity")
    st.caption("Adjust your monthly SIP amount for loans or other temporary cashflow changes.")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Add SIP Effect", use_container_width=True):
            st.session_state.effects.append({
                'id': len(st.session_state.effects),
                'type': 'Manual', # Default
                'start_date': datetime(2030, 1, 1),
                'end_date': datetime(2250, 1, 1),
                'monthly_amount': -30000,
                'pv': 5000000,
                'interest_rate': 8.5,
                'inflation_rate': 6.0
            })
            st.rerun()

    if len(st.session_state.effects) > 0:
        for i, effect in enumerate(st.session_state.effects):
            with st.expander(f"Effect {i+1}", expanded=True):
                effect.setdefault('type', 'Manual')
                type_col, _ = st.columns([1, 3])
                with type_col:
                    effect['type'] = st.selectbox("Type", ["Manual", "Loan EMI"], key=f"effect_type_{i}")
                
                if effect['type'] == "Manual":
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                    with col1:
                        effect['start_date'] = date_text_input("Start", value=effect['start_date'], key=f"eff_s_{i}")
                    with col2:
                        effect['end_date'] = date_text_input("End", value=effect['end_date'], key=f"eff_e_{i}")
                    with col3:
                        effect['monthly_amount'] = st.number_input("Amount (₹) PV", value=int(effect['monthly_amount']), step=1000, key=f"eff_a_{i}", help="Present Value - will be adjusted for inflation")
                    with col4:
                        effect['inflation_rate'] = st.number_input("Inflation (%)", value=float(effect.get('inflation_rate', 6.0)), step=0.1, key=f"eff_inf_{i}")
                    with col5:
                        if st.button("🗑️", key=f"del_eff_{i}"):
                            st.session_state.effects.pop(i)
                            st.rerun()
                else: 
                    c1, c2, c3 = st.columns(3)
                    with c1: effect['pv'] = st.number_input("Loan PV", value=effect.get('pv', 5000000), step=100000, key=f"lpv_{i}")
                    with c2: effect['interest_rate'] = st.number_input("Interest %", value=effect.get('interest_rate', 8.5), step=0.1, key=f"lint_{i}")
                    with c3: effect['inflation_rate'] = st.number_input("Inflation %", value=effect.get('inflation_rate', 6.0), step=0.1, key=f"linf_{i}")
                    c4, c5, c6 = st.columns([2, 2, 2])
                    with c4: 
                        effect['start_date'] = date_text_input("Start", value=effect['start_date'], key=f"lst_{i}")
                    with c5: 
                        effect['end_date'] = date_text_input("End", value=effect['end_date'], key=f"lend_{i}")
                    
                    calculated_emi = calculate_emi(effect['pv'], pd.Timestamp(effect['start_date']), pd.Timestamp(effect['end_date']), effect['interest_rate'], effect['inflation_rate'], pd.Timestamp(current_date))
                    effect['monthly_amount'] = -calculated_emi
                    
                    with c6:
                        st.metric("EMI", logic.format_inr(calculated_emi))
                    if st.button("🗑️", key=f"del_leff_{i}"):
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
            core_return = st.number_input("Return (%)", value=12.0, step=0.1, key="core_return")
            core_tax = st.number_input("Tax (%)", value=12.5, step=0.1, key="core_tax")
    
    goal_return = 0.0
    goal_tax = 0.0

    st.divider()

    # Generate Output Button
    if st.button("🚀 Find Earliest Retirement Date", type="primary", use_container_width=True):

        # Validate goals against death_date before running
        invalid_goals = [
            g['name'] or f"Goal {i+1}"
            for i, g in enumerate(st.session_state.goals)
            if pd.Timestamp(g['maturity_date']) >= death_date
        ]
        if invalid_goals:
            st.error(f"Cannot run: the following goals have maturity dates at or after your target lifetime ({death_date.strftime('%b %Y')}): {', '.join(invalid_goals)}. Please adjust them.")
            st.stop()

        # Create input_variables dictionary
        # Prepare Expense Streams Data
        mapped_expense_streams = []
        for s in st.session_state.expense_streams:
            adjs = []
            for a in s['adjustments']:
                entry = {
                    'start_date': pd.Timestamp(a['start_date']),
                    'end_date': pd.Timestamp(a['end_date'])
                }
                if a.get('type') == 'Fixed Amount':
                    entry['amount'] = float(a['value'])
                else:
                    entry['percentage'] = float(a['value'])
                adjs.append(entry)
                
            mapped_expense_streams.append({
                'name': s['name'],
                'amount': int(s['amount']),
                'frequency': int(s['frequency']),
                'inflation': float(s['inflation']),
                'post_retirement_change': float(s['post_retirement_change']),
                'adjustments': adjs
            })

        # Create input_variables dictionary
        input_variables = {
            'current_date': pd.Timestamp(current_date),
            'current_age': int(current_age),
            'target_lifetime': int(target_lifetime),
            'current_corpus': int(current_corpus),
            'current_sip': int(current_sip),
            'yearly_sip_step_up_%': float(yearly_sip_step_up),
            'stepup_date_month': stepup_month,
            'stepup_date_day': stepup_day,
            'sip_adjustments': [
                {'start_date': pd.Timestamp(adj['start_date']), 'end_date': pd.Timestamp(adj['end_date']), 'percentage': float(adj['percentage'])}
                for adj in st.session_state.sip_adjustments
            ],
            'expense_streams': mapped_expense_streams,
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
                    'monthly_amount': int(effect['monthly_amount']),
                    'inflation_rate': float(effect.get('inflation_rate', 0.0))
                }
                for effect in st.session_state.effects
            ]
        }
        
        instrument_params = {
            'hybrid': {'return': hybrid_return / 100, 'tax': hybrid_tax / 100},
            'debt': {'return': debt_return / 100, 'tax': debt_tax / 100},
            'goal': {'return': goal_return / 100, 'tax': goal_tax / 100},
            'core_corpus': {'return': core_return / 100, 'tax': core_tax / 100}
        }
        
        all_glide_paths = st.session_state.standard_glide_paths.copy()
        all_glide_paths.update(st.session_state.custom_glide_paths)

        with st.spinner("Simulating... This may take a moment."):
            result = logic.find_retirement_date(input_variables, instrument_params, all_glide_paths)
            
        if result:
            ret_month, ret_year = result
            retirement_date = pd.Timestamp(year=ret_year, month=ret_month, day=1)
            
            # Calculate Age at Retirement
            # Simplistic age calc: Current Age + Years Passed
            years_passed = (retirement_date - pd.Timestamp(current_date)).days / 365.25
            retirement_age = current_age + years_passed
            
            st.success(f"🎉 You can retire on {retirement_date.strftime('%B %Y')}!")
            st.metric("Retirement Age", f"{retirement_age:.1f} Years")
            
            # Rerun simulation with found date to get details (moved up to access comprehensive_df for summary)
            success, final_trans_df, _, expense_movements_df, goal_dfs, comprehensive_df = logic.run_simulation(input_variables, retirement_date, instrument_params, all_glide_paths)
            
            if success:
                # --- Retirement Summary Section ---
                st.subheader("💰 Financial Summary at Retirement")
                st.caption(f"Snapshot of your wealth distribution on {retirement_date.strftime('%B %Y')}")
                
                # Find the row closest to retirement date in comprehensive_df
                comprehensive_df['Date'] = pd.to_datetime(comprehensive_df['Date'])
                ret_data = comprehensive_df[comprehensive_df['Date'] >= retirement_date].head(1)
                
                if not ret_data.empty:
                    ret_row = ret_data.iloc[0]
                    
                    # Core Corpus Value
                    core_val = ret_row.get('Core Corpus Value', 0)
                    debt_pool_val = ret_row.get('Expense Debt Pool Value', 0)
                    hybrid_pool_val = ret_row.get('Expense Hybrid Pool Value', 0)
                    
                    # Goal-specific columns
                    goal_totals = {'debt': 0, 'hybrid': 0}
                    goal_details = []
                    for goal_info in input_variables['goals']:
                        gname = goal_info['name']
                        g_debt = ret_row.get(f'{gname} Debt Value', 0)
                        g_hybrid = ret_row.get(f'{gname} Hybrid Value', 0)
                        goal_totals['debt'] += g_debt
                        goal_totals['hybrid'] += g_hybrid
                        if g_debt > 0 or g_hybrid > 0:
                            goal_details.append({'Goal': gname, 'Debt': g_debt, 'Hybrid': g_hybrid, 'Total': g_debt + g_hybrid})
                    
                    # Total Wealth
                    total_wealth = core_val + debt_pool_val + hybrid_pool_val + goal_totals['debt'] + goal_totals['hybrid']
                    
                    # Display Summary
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Core Corpus", logic.format_inr(core_val))
                    col_b.metric("Expense Debt Pool", logic.format_inr(debt_pool_val))
                    col_c.metric("Expense Hybrid Pool", logic.format_inr(hybrid_pool_val))
                    
                    if goal_details:
                        st.markdown("**Goal Pools at Retirement:**")
                        goal_summary_df = pd.DataFrame(goal_details)
                        # Format for display
                        goal_summary_df['Debt'] = goal_summary_df['Debt'].apply(logic.format_inr)
                        goal_summary_df['Hybrid'] = goal_summary_df['Hybrid'].apply(logic.format_inr)
                        goal_summary_df['Total'] = goal_summary_df['Total'].apply(logic.format_inr)
                        st.dataframe(goal_summary_df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No goal-specific pools active at retirement date.")
                    
                    st.divider()
                    st.metric("💎 Total Wealth at Retirement", logic.format_inr(total_wealth))
                    st.divider()
            
                # Logic to generate daily value for chart
                # run_simulation returns (success, final_trans_df)
                # We need daily values. logic.run_simulation doesn't return full dict anymore in main_v2? 
                
                # 1. Generate NAV - cap at death_date for consistent display
                nav_df = logic.generate_pseudo_nav(input_variables['current_date'], death_date, instrument_params['core_corpus']['return'])

                # 2. Daily Value
                daily_value_df = logic.calculate_daily_value(final_trans_df, nav_df)

                # 3. Cap the chart data at death_date to avoid misleading uptick after expenses stop
                daily_value_df = daily_value_df[daily_value_df['Date'] <= death_date]

                st.subheader(f"Projected Core Corpus (Until Age {int(target_lifetime)})")
                st.line_chart(daily_value_df, x='Date', y='current_value')

                with st.expander("Show Daily Core Corpus Data"):
                    st.dataframe(daily_value_df)

                # Corpus remaining at target lifetime
                st.subheader(f"Wealth at Age {int(target_lifetime)}")
                death_data = comprehensive_df[pd.to_datetime(comprehensive_df['Date']) <= death_date]
                if not death_data.empty:
                    death_row = death_data.iloc[-1]
                    death_core_val = death_row.get('Core Corpus Value', 0)
                    death_debt_val = death_row.get('Expense Debt Pool Value', 0)
                    death_hybrid_val = death_row.get('Expense Hybrid Pool Value', 0)
                    death_pool_total = death_debt_val + death_hybrid_val
                    death_total = death_core_val + death_pool_total

                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric(f"Core Corpus", logic.format_inr(death_core_val))
                    d2.metric("Debt Pool Buffer", logic.format_inr(death_debt_val))
                    d3.metric("Hybrid Pool Buffer", logic.format_inr(death_hybrid_val))
                    d4.metric("Total Estate", logic.format_inr(death_total))
                    st.caption(
                        "The Debt and Hybrid pool buffers represent expenses pre-funded "
                        "5 years beyond your target lifetime — a conservative safety reserve."
                    )

                st.subheader("Goals Status")
                for goal in input_variables['goals']:
                    goal_fv = logic.future_value(
                        goal['downpayment_present_value'],
                        goal['rate_for_future_value%'] / 100,
                        input_variables['current_date'],
                        goal['maturity_date']
                    )
                    maturity_str = goal['maturity_date'].strftime('%b %Y')
                    st.success(f"✅ {goal['name']} — FV: {logic.format_inr(goal_fv)} @ {maturity_str} — Fully Funded")
                    
                st.info("Since the simulation succeeded, all goals are fully funded according to their glide paths.")
                
                # Display Dataframes requested by user
                st.divider()
                st.header("📋 Simulation Details")
                
                st.subheader("Consolidated Core Transactions")
                st.dataframe(final_trans_df)
                
                st.subheader("Expense Pool Movements (Debt & Hybrid)")
                st.dataframe(expense_movements_df)
                
                st.subheader("Comprehensive Month-by-Month View")
                st.caption("Includes Core Corpus, Expense Pools, Goal Pools, Net SIP, and Net Expenses.")
                st.dataframe(comprehensive_df)

                st.subheader("Goal Specific Details")
                for goal_name, goal_df in goal_dfs.items():
                    # Find matching goal to compute FV
                    matching_goal = next((g for g in input_variables['goals'] if g['name'] == goal_name), None)
                    if matching_goal:
                        goal_fv = logic.future_value(
                            matching_goal['downpayment_present_value'],
                            matching_goal['rate_for_future_value%'] / 100,
                            input_variables['current_date'],
                            matching_goal['maturity_date']
                        )
                        maturity_str = matching_goal['maturity_date'].strftime('%b %Y')
                        expander_label = f"Goal: {goal_name} — FV: {logic.format_inr(goal_fv)} @ {maturity_str}"
                    else:
                        expander_label = f"Goal: {goal_name}"
                        
                    with st.expander(expander_label):
                        st.dataframe(goal_df)
                
            else:
                st.error("Simulation failed upon re-verification. Please report this bug.")
                
        else:
            st.error(f"Cannot retire before age {int(target_lifetime)} with the current configuration.")
            
            # Diagnose Failure
            st.subheader("Diagnosis: Why can't you retire?")
            with st.spinner("Analyzing failure..."):
                # Run simulation assuming retirement postponed to death_date (target lifetime age)
                # This maximises savings and delays expenses to show best-case failure point
                diag_retirement_date = death_date

                sim_msg = f"Simulating scenario where retirement is postponed to **{diag_retirement_date.strftime('%B %Y')}** (age {int(target_lifetime)}) to maximize savings and delay expenses."
                st.info(sim_msg)
                
                success, _, failure_details, _, _, _ = logic.run_simulation(input_variables, diag_retirement_date, instrument_params, all_glide_paths)
                
                if not success and failure_details:
                    fail_date = failure_details['date']
                    fail_amount = failure_details['amount']
                    fail_desc = failure_details['description']
                    
                    st.warning(f"Projected Corpus depletion date: {fail_date.strftime('%B %Y')}")
                    
                    st.markdown(f"""
                    ### First Failure Event
                    - **Date**: {fail_date.strftime('%d %B %Y')}
                    - **Reason**: {fail_desc}
                    - **Shortfall Amount**: {logic.format_inr(fail_amount)}
                    """)
                    
                    if "Monthly Expense" in fail_desc:
                        st.write("Your monthly expenses are draining the corpus. Consider reducing expenses or increasing SIP.")
                    else:
                        st.write("A specific goal or transfer caused the shortfall. Consider adjusting this goal's value or date.")
                else:
                     st.write("Corpus seems to survive with immediate retirement? This is unexpected if 'find_retirement_date' failed. It might be due to 100-year projection limit.")

        st.divider()
        with st.expander("View Input Configuration"):
            def make_serializable(obj):
                if isinstance(obj, (pd.Timestamp, date, datetime)):
                    return obj.strftime('%Y-%m-%d')
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [make_serializable(i) for i in obj]
                return obj
            st.json(make_serializable(input_variables))

if __name__ == "__main__":
    main()
