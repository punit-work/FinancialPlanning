import os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# All Date columns / Timestamps must use a single resolution to avoid
# pandas merge_asof dtype-mismatch errors across versions.  We standardise
# on nanosecond resolution (datetime64[ns]) everywhere.
# ---------------------------------------------------------------------------
_NS_DTYPE = "datetime64[ns]"

def _ensure_date_ns(df):
    """Cast the 'Date' column of *df* to datetime64[ns] **in-place** and return df."""
    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(_NS_DTYPE)
    return df

def _ts(val):
    """Return a pd.Timestamp guaranteed to be nanosecond resolution."""
    return pd.Timestamp(val).as_unit("ns")
# import math
# import copy

class TaxLot:
    def __init__(self, date, units, purchase_price_per_unit):
        self.date = pd.Timestamp(date)
        self.units = float(units)
        self.purchase_price = float(purchase_price_per_unit)
        self.purchase_val = self.units * self.purchase_price

    def current_value(self, current_nav):
        return self.units * current_nav

class InvestmentPool:
    def __init__(self, name, stcg_tax, ltcg_tax):
        self.name = name
        self.stcg_tax = stcg_tax
        self.ltcg_tax = ltcg_tax
        self.lots = [] # List of TaxLot objects

    def _get_tax_rate(self, lot_date, redemption_date):
        holding_days = (pd.Timestamp(redemption_date) - pd.Timestamp(lot_date)).days
        return self.stcg_tax if holding_days <= 365 else self.ltcg_tax

    def invest(self, date, amount, nav, description="Investment"):
        if amount <= 0: return None
        units = amount / nav
        new_lot = TaxLot(date, units, nav)
        self.lots.append(new_lot)
        return {
            'Date': date, 'Amount': amount, 'NAV': nav, 'units': units,
            'Description': description, 'tax': 0, 'fully_funded': True, 'shortfall': 0, 'source': 'Investment',
            'Pool': self.name
        }

    def get_market_value(self, nav):
        return sum(lot.units for lot in self.lots) * nav

    def get_unrealized_tax(self, nav, as_of_date=None):
        total_tax = 0
        for lot in self.lots:
            gain_per_unit = nav - lot.purchase_price
            if gain_per_unit > 0:
                rate = self._get_tax_rate(lot.date, as_of_date) if as_of_date is not None else self.ltcg_tax
                total_tax += gain_per_unit * lot.units * rate
        return total_tax

    def redeem_net_amount(self, date, target_net, nav, description="Withdrawal"):
        # We need to withdraw enough units such that (Value - Tax) = target_net
        # Since tax depends on which lots are sold (FIFO), this is iterative or requires handling lot by lot.
        
        needed_net = target_net
        total_gross_withdrawn = 0
        total_tax = 0
        total_units = 0
        
        lots_to_remove = []
        lots_updated = {} # index -> new_units
        
        trans_details = []

        # Iterate through lots FIFO
        for i, lot in enumerate(self.lots):
            if needed_net <= 1e-4: break
            
            # Max we can get from this lot
            curr_val = lot.current_value(nav)
            gain_per_unit = nav - lot.purchase_price
            tax_per_unit = max(0, gain_per_unit * self._get_tax_rate(lot.date, date))
            net_per_unit = nav - tax_per_unit
            
            # Check if this lot covers the remainder
            max_net_from_lot = lot.units * net_per_unit
            
            if max_net_from_lot <= needed_net:
                # Consume entire lot
                units_to_sell = lot.units
                gross_amt = curr_val
                tax_amt = units_to_sell * tax_per_unit
                
                needed_net -= (gross_amt - tax_amt)
                total_gross_withdrawn += gross_amt
                total_tax += tax_amt
                total_units += units_to_sell
                lots_to_remove.append(i)
                
            else:
                # Partial lot
                units_to_sell = needed_net / net_per_unit
                gross_amt = units_to_sell * nav
                tax_amt = units_to_sell * tax_per_unit
                
                needed_net = 0
                total_gross_withdrawn += gross_amt
                total_tax += tax_amt
                total_units += units_to_sell
                
                # Update lot remaining units
                lots_updated[i] = lot.units - units_to_sell

        # Apply updates
        # Process updates first
        for i, new_units in lots_updated.items():
            self.lots[i].units = new_units
            
        # Process removals (reverse order to keep indices valid)
        for i in sorted(lots_to_remove, reverse=True):
            self.lots.pop(i)

        fully_funded = (needed_net <= 1.0) # Floating point tolerance
        
        return {
            'Date': date, 'Amount': -total_gross_withdrawn, 'NAV': nav,
            'units': -total_units, 'Description': description,
            'tax': total_tax, 'fully_funded': fully_funded, 
            'shortfall': needed_net,
            'net_received': total_gross_withdrawn - total_tax,
            'Pool': self.name
        }

    def redeem_gross_amount(self, date, target_gross, nav, description="Withdrawal Gross"):
        # Simpler: just sell units to meet target gross
        needed_gross = target_gross
        total_gross_withdrawn = 0
        total_tax = 0
        total_units = 0
        
        lots_to_remove = []
        lots_updated = {}
        
        for i, lot in enumerate(self.lots):
            if needed_gross <= 1e-4: break
            
            curr_val = lot.current_value(nav)
            
            if curr_val <= needed_gross:
                # Consume entire lot
                units_to_sell = lot.units
                gross_amt = curr_val
                gain = gross_amt - lot.purchase_val
                tax = max(0, gain * self._get_tax_rate(lot.date, date))

                needed_gross -= gross_amt
                total_gross_withdrawn += gross_amt
                total_tax += tax
                total_units += units_to_sell
                lots_to_remove.append(i)

            else:
                # Partial lot
                fraction = needed_gross / curr_val
                units_to_sell = lot.units * fraction
                gross_amt = needed_gross

                purchase_cost_for_part = lot.purchase_val * fraction
                gain = gross_amt - purchase_cost_for_part
                tax = max(0, gain * self._get_tax_rate(lot.date, date))
                
                needed_gross = 0
                total_gross_withdrawn += gross_amt
                total_tax += tax
                total_units += units_to_sell
                
                lots_updated[i] = lot.units - units_to_sell
        
        for i, new_units in lots_updated.items():
            self.lots[i].units = new_units
        for i in sorted(lots_to_remove, reverse=True):
            self.lots.pop(i)
            
        fully_funded = (needed_gross <= 1.0)
        
        return {
            'Date': date, 'Amount': -total_gross_withdrawn, 'NAV': nav,
            'units': -total_units, 'Description': description,
            'tax': total_tax, 'fully_funded': fully_funded, 
            'shortfall': needed_gross,
            'net_received': total_gross_withdrawn - total_tax,
            'Pool': self.name
        }

def calculate_corpus_required_for_future_expense(expense_amount, years_to_expense, rate_of_return, tax_rate):
    # Formula: P = E / [ (1+r)^t(1-tax) + tax ]
    # Where E is expense, r is rate, t is time in years
    
    growth_factor = (1 + rate_of_return) ** years_to_expense
    denominator = growth_factor * (1 - tax_rate) + tax_rate
    required_corpus = expense_amount / denominator
    return required_corpus


# --- Helper Functions from main.py ---

def format_inr(amount):
    amount = round(float(amount), 2)
    integer, decimal = f"{amount:.2f}".split(".")

    if len(integer) > 3:
        last3 = integer[-3:]
        rest = integer[:-3]
        rest = ",".join([rest[max(i-2, 0):i] for i in range(len(rest), 0, -2)][::-1])
        integer = rest + "," + last3

    return f"₹{integer}.{decimal}"

def future_value(present_value, inflation_rate, current_date, future_date):
    # Time difference in years (actual days / 365.25)
    years = (future_date - current_date).days / 365.25
    # Future value calculation
    fv = present_value * ((1 + inflation_rate) ** years)
    return round(fv, 2)

def calculate_stepup_occurrences(start_date, end_date, month, day):
    if month is None or day is None:
        return 0
    
    count = 0
    current_year = start_date.year
    
    # Check if we need to start from the next year
    try:
        first_stepup = pd.Timestamp(year=current_year, month=int(month), day=int(day))
    except ValueError:
        current_year += 1
        first_stepup = pd.Timestamp(year=current_year, month=int(month), day=int(day))
    
    if first_stepup < start_date:
        current_year += 1
    
    while True:
        try:
            stepup_date = pd.Timestamp(year=current_year, month=int(month), day=int(day))
            if stepup_date > end_date:
                break
            if stepup_date >= start_date:
                count += 1
            current_year += 1
        except ValueError:
            current_year += 1
            continue
    
    return count

def generate_pseudo_nav(start_date, end_date, rate_of_return):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    annual_rate = rate_of_return
    daily_rate = (1 + annual_rate) ** (1/365) - 1
    days_elapsed = np.arange(len(date_range))
    nav_values = 100 * (1 + daily_rate) ** days_elapsed
    
    pseudo_nav_df = pd.DataFrame({
        'Date': date_range,
        'nav': nav_values
    })
    return _ensure_date_ns(pseudo_nav_df)

# --- Core Calculation Functions ---

def calculate_goal_cashflows(input_df, end_date, goal_value_post_tax, instrument_params, input_variables):
    current_date = input_variables['current_date']
    df = input_df.copy()
    end_date = pd.Timestamp(end_date)
    df['place'] = df['place'].str.lower()
    
    df['inflow_date'] = df['years from inflow till end'].apply(
        lambda years: end_date - relativedelta(years=int(years))
    )
    
    df['outflow_date'] = df['years from outflow till end'].apply(
        lambda x: end_date - relativedelta(years=int(x)) if pd.notna(x) else pd.NaT
    )

    df['inflow_date'] = df['inflow_date'].astype(_NS_DTYPE)
    df['outflow_date'] = pd.to_datetime(df['outflow_date']).astype(_NS_DTYPE)

    df[['inflow_date', 'outflow_date']] = df[['inflow_date', 'outflow_date']].mask(
        df[['inflow_date', 'outflow_date']] < current_date,
        current_date
    )
    
    df['goal_value_post_tax'] = goal_value_post_tax
    df['inflow_amount'] = 0.0
    id_to_idx = {row['id']: idx for idx, row in df.iterrows()}
    
    def calculate_required_inflow(target_post_tax, annual_return, tax_rate, years):
        if years == 0:
            return target_post_tax
        growth_factor = (1 + annual_return) ** years
        multiplier = growth_factor * (1 - tax_rate) + tax_rate
        return target_post_tax / multiplier
    
    def process_chain(goal_row_id):
        current_id = goal_row_id
        current_idx = id_to_idx[current_id]
        current_row = df.loc[current_idx]
        
        target_amount = current_row['goal_value_post_tax'] * (current_row['% of goal value'] / 100)
        df.at[current_idx, 'inflow_amount'] = target_amount
        
        while True:
            current_idx = id_to_idx[current_id]
            current_row = df.loc[current_idx]
            inflow_from = current_row['inflow_from']
            
            if inflow_from == 'core corpus':
                break
            
            source_idx = id_to_idx[inflow_from]
            source_row = df.loc[source_idx]
            
            inflow_date = source_row['inflow_date']
            outflow_date = current_row['inflow_date']
            years = (outflow_date - inflow_date).days / 365.25
            
            place = source_row['place'].lower()
            params = instrument_params.get(place, {'return': 0.0, 'stcg_tax': 0.0, 'ltcg_tax': 0.0})
            tax_rate = params['stcg_tax'] if years <= 1 else params['ltcg_tax']

            target_for_source = df.at[current_idx, 'inflow_amount']
            required_inflow = calculate_required_inflow(
                target_for_source, params['return'], tax_rate, years
            )
            
            df.at[source_idx, 'inflow_amount'] = required_inflow
            current_id = inflow_from
    
    goal_rows = df[df['place'] == 'goal']
    for _, goal_row in goal_rows.iterrows():
        process_chain(goal_row['id'])
    
    df['inflow_amount'] = df['inflow_amount'].round(2)
    df['total_outflow_amount'] = 0.0
    df['tax_out_of_outflow'] = 0.0
    
    for idx, row in df.iterrows():
        if row['place'] == 'goal':
            df.at[idx, 'total_outflow_amount'] = pd.NA
            df.at[idx, 'tax_out_of_outflow'] = pd.NA
            continue
        
        place = row['place'].lower()
        params = instrument_params.get(place, {'return': 0.0, 'stcg_tax': 0.0, 'ltcg_tax': 0.0})

        if pd.notna(row['outflow_date']):
            years = (row['outflow_date'] - row['inflow_date']).days / 365.25
            principal = row['inflow_amount']
            total_outflow = principal * ((1 + params['return']) ** years)
            gains = total_outflow - principal
            tax_rate = params['stcg_tax'] if years <= 1 else params['ltcg_tax']
            tax = gains * tax_rate
            
            df.at[idx, 'total_outflow_amount'] = round(total_outflow, 2)
            df.at[idx, 'tax_out_of_outflow'] = round(tax, 2)
        else:
            df.at[idx, 'total_outflow_amount'] = pd.NA
            df.at[idx, 'tax_out_of_outflow'] = pd.NA
    
    output_columns = [
        'id', 'place', 'inflow_date', 'outflow_date', 'inflow_from',
        'outflow_to', '% of goal value', 'goal_value_post_tax', 'inflow_amount',
        'total_outflow_amount', 'tax_out_of_outflow'
    ]
    return df[output_columns]

def calculate_sip_cashflows(input_variables, last_goal_date, retirement_date):
    current_date = input_variables['current_date']
    current_sip = input_variables['current_sip']
    yearly_step_up = input_variables['yearly_sip_step_up_%'] / 100
    effects_on_cashflows = input_variables['effects_on_cashflows']
    
    stepup_date_month = input_variables.get('stepup_date_month', None)
    stepup_date_day = input_variables.get('stepup_date_day', None)
    sip_adjustments = input_variables.get('sip_adjustments', [])
    
    # We generate SIPs only until retirement date or last_goal_date, 
    # but strictly SIP should stop at retirement_date according to user request.
    # We will generate up to max of (retirement, last_goal) but set SIP amount to 0 after retirement.
    
    end_date = max(last_goal_date, retirement_date)
    date_range = pd.date_range(start=current_date, end=end_date, freq='MS')

    df = _ensure_date_ns(pd.DataFrame({'Date': date_range}))
    sip_amounts = []

    for i, date in enumerate(date_range):
        # Stop SIPs at retirement
        if date >= retirement_date:
            sip_amounts.append(0.0)
            continue
            
        if stepup_date_month is not None and stepup_date_day is not None:
            years_elapsed = calculate_stepup_occurrences(
                current_date, date, stepup_date_month, stepup_date_day
            )
        else:
            years_elapsed = i // 12
        
        base_sip_amount = current_sip * ((1 + yearly_step_up) ** years_elapsed)
        final_sip_amount = base_sip_amount
        
        for adjustment in sip_adjustments:
            if adjustment['start_date'] <= date <= adjustment['end_date']:
                final_sip_amount = base_sip_amount * (adjustment['percentage'] / 100)
                break
        
        sip_amounts.append(final_sip_amount)
    
    df['SIP amount'] = sip_amounts
    df['adjustment amount'] = 0.0
    
    # Effects on cashflows apply probably during working years? 
    # Or should we assume they are distinct from SIPs?
    # Usually "effects on cashflows" adjust the SIP capacity.
    # So they should also stop at retirement if they are salary adjustments.
    # Assuming they stop at retirement for safety unless explicitly handled.
    
    for effect in effects_on_cashflows:
        start_date = effect['start_date']
        end_date = effect['end_date']
        # Cap end_date at retirement for cashflow effects
        effective_end_date = min(end_date, retirement_date)
        inflation_rate = effect.get('inflation_rate', 0.0) / 100.0

        if start_date < effective_end_date:
            mask = (df['Date'] >= start_date) & (df['Date'] < effective_end_date)
            if inflation_rate != 0:
                # Amount is present value relative to current_date, adjust for inflation
                for idx in df[mask].index:
                    years_elapsed = (df.loc[idx, 'Date'] - current_date).days / 365.25
                    adjusted_amount = effect['monthly_amount'] * ((1 + inflation_rate) ** years_elapsed)
                    df.loc[idx, 'adjustment amount'] += adjusted_amount
            else:
                df.loc[mask, 'adjustment amount'] += effect['monthly_amount']
    
    df['net sip amount'] = df['SIP amount'] + df['adjustment amount']
    return df

def calculate_expenses_cashflows(input_variables, retirement_date, simulation_end_date=None):
    current_date = input_variables['current_date']

    streams = input_variables.get('expense_streams', [])

    # Use provided end date or default to 100 years from current date
    if simulation_end_date is None:
        simulation_end_date = current_date + pd.DateOffset(years=100)
    
    # We need a master timeline to aggregate onto
    master_date_range = pd.date_range(start=current_date, end=simulation_end_date, freq='MS')
    master_df = _ensure_date_ns(pd.DataFrame({'Date': master_date_range}))
    master_df['Expense Amount'] = 0.0

    for stream in streams:
        name = stream.get('name', 'Expense')
        amount = stream.get('amount', 0)
        freq_months = max(1, int(stream.get('frequency', 1))) # Minimum 1 month
        inflation = stream.get('inflation', 0) / 100.0
        ret_change = stream.get('post_retirement_change', 0) / 100.0
        adjustments = stream.get('adjustments', [])
        
        # Generate dates for this stream by stepping through months
        stream_dates = []
        d = current_date
        while d <= simulation_end_date:
            stream_dates.append(d)
            d = d + relativedelta(months=freq_months)
        
        # Calculate values
        stream_values = []
        for date in stream_dates:
            years_elapsed = (date - current_date).days / 365.25
            
            # 1. Inflation
            val = amount * ((1 + inflation) ** years_elapsed)
            
            # 2. Retirement Step-up
            if date >= retirement_date:
                val = val * (1 + ret_change)
            
            # 3. Adjustments (Specific to this stream)
            for adj in adjustments:
                # Adjustments in stream are likely dicts with start, end, percentage OR amount
                # Ensure date types match
                s_date = pd.Timestamp(adj['start_date'])
                e_date = pd.Timestamp(adj['end_date'])
                
                if s_date <= date <= e_date:
                    if 'percentage' in adj and adj['percentage'] is not None:
                        val = val * (adj['percentage'] / 100.0)
                    elif 'amount' in adj and adj['amount'] is not None:
                        val = val + adj['amount']
                    
                    # Assuming non-overlapping or sequential application? 
                    # If overlapping, order matters. Code breaks after first match.
                    break
            
            stream_values.append(val)
            
        # Create DF for this stream
        s_df = _ensure_date_ns(pd.DataFrame({'Date': stream_dates, 'Amount': stream_values}))
        
        # Resample to Monthly (Summing) to match Master DF granularity
        # We align to Month Start for consistency with master
        s_df['YearMonth'] = s_df['Date'].dt.to_period('M')
        monthly_sum = s_df.groupby('YearMonth')['Amount'].sum().reset_index()
        monthly_sum['Date'] = monthly_sum['YearMonth'].dt.to_timestamp() # Defaults to month start
        
        # Merge into accumulator
        # We can just concat and group later or merge iteratively. 
        # Since we want a single 'Expense Amount' column in master, let's merge.
        
        # Align dates exactly to master (which is Month Start)
        # Note: to_timestamp() gives Month Start (1st day).
        
        master_df = master_df.merge(monthly_sum[['Date', 'Amount']], on='Date', how='left').fillna(0)
        master_df['Expense Amount'] += master_df['Amount']
        master_df = master_df.drop(columns=['Amount'])
        
    master_df['Net Expense Amount'] = master_df['Expense Amount'].apply(lambda x: max(0, x))

    return master_df

def calculate_passive_income_cashflows(config, retirement_date, simulation_end_date=None):
    """
    Generates a monthly passive income series starting from retirement_date.
    Each stream's today's-rupee amount is grown to retirement at pre_retirement_growth,
    then continues growing at post_retirement_growth from there onward.
    """
    current_date = config['current_date']
    streams = config.get('passive_income_streams', [])
    retirement_date = pd.Timestamp(retirement_date)

    if simulation_end_date is None:
        simulation_end_date = current_date + pd.DateOffset(years=100)

    master_date_range = pd.date_range(start=retirement_date, end=simulation_end_date, freq='MS')
    master_df = _ensure_date_ns(pd.DataFrame({'Date': master_date_range}))
    master_df['Passive Income Amount'] = 0.0

    if not streams:
        return master_df

    years_to_retirement = (retirement_date - current_date).days / 365.25
    years_post_ret = (master_df['Date'] - retirement_date).dt.days / 365.25

    for stream in streams:
        amount = stream.get('amount', 0)
        pre_growth = stream.get('pre_retirement_growth', 0) / 100.0
        post_growth = stream.get('post_retirement_growth', 0) / 100.0
        end_date = stream.get('end_date', None)

        # Grow from today to retirement at pre-retirement rate
        amount_at_retirement = amount * ((1 + pre_growth) ** years_to_retirement)

        # Grow post-retirement at post-retirement rate
        stream_values = amount_at_retirement * ((1 + post_growth) ** years_post_ret)

        # Zero out after end_date if specified
        if end_date is not None:
            end_ts = pd.Timestamp(end_date)
            stream_values = stream_values.where(master_df['Date'] <= end_ts, 0.0)

        master_df['Passive Income Amount'] += stream_values

    return master_df


def get_withdrawl_df(goal_dfs):
    results = []
    for name, df in goal_dfs.items():
        # Filter for 'core corpus' withdrawals
        for _, row in df[df['inflow_from']=='core corpus'].copy(deep=True).sort_values(by='inflow_date').iterrows():
            results.append({
                'Date': row['inflow_date'],
                'Amount': row['inflow_amount'],
                'Description': f'Moving to {row["place"]} for {name} goal.'
            })
    return pd.DataFrame(results)

def create_sip_trans(nav_df, sip_df, input_variables, retirement_date):
    trans = []
    current_corpus = input_variables['current_corpus']
    current_date = input_variables['current_date']
    
    # Initial Corpus
    nav = nav_df[nav_df['Date'] == current_date]['nav'].iloc[-1]
    units = current_corpus / nav
    trans.append({
        'Date': current_date, 'Amount': current_corpus, 'NAV': nav, 'units': units, 'Description': 'Current Corpus'
    })
    
    # SIP Inflows
    for _, row in sip_df.iterrows():
        amount = row['net sip amount']
        date = row['Date']
        
        # Look up NAV
        matches = nav_df[nav_df['Date'] <= date]
        if not matches.empty:
            nav = matches['nav'].iloc[-1]
        else:
            nav = nav_df['nav'].iloc[-1] # fallback
            
        if amount > 0:
            units = amount / nav
            trans.append({
                'Date': date, 'Amount': amount, 'NAV': nav, 'units': units, 'Description': 'Monthly inflow'
            })
            
    return pd.DataFrame(trans)

def add_withdrawls_to_trans(sip_trans_df, withdrawls_df, nav_df, instrument_params):
    updated_trans_df = sip_trans_df.copy(deep=True)
    withdrawal_transactions = []
    
    # Combine withdrawals from goals and post-retirement expenses
    # Assume withdrawls_df has both
    
    # Sort withdrawals by date
    withdrawls_df = withdrawls_df.sort_values('Date').reset_index(drop=True)
    
    for _, row in withdrawls_df.iterrows():
        amount = row['Amount']
        date = row['Date']
        description = row['Description']
        
        # Get NAV
        matches = nav_df[nav_df['Date'] <= date]
        if not matches.empty:
            current_nav = matches['nav'].iloc[-1]
        else:
            current_nav = nav_df['nav'].iloc[0] # Should not happen if nav covers range
            
        # Get available units up to this date
        available_trans_df = updated_trans_df[updated_trans_df['Date'] <= date].copy()
        
        if available_trans_df.empty:
             withdrawal_transactions.append({
                'Date': date, 'Amount': -amount, 'NAV': current_nav, 'units': -amount/current_nav,
                'Description': description, 'tax': 0, 'fully_funded': False, 'shortfall': amount
            })
             continue
             
        # Calculate Taxes and Liquidation
        cc_stcg = instrument_params['core_corpus']['stcg_tax']
        cc_ltcg = instrument_params['core_corpus']['ltcg_tax']
        available_trans_df['current_value'] = available_trans_df['units'] * current_nav
        available_trans_df['gains'] = available_trans_df['current_value'] - available_trans_df['Amount']
        available_trans_df['holding_days'] = (date - available_trans_df['Date']).dt.days
        available_trans_df['applicable_tax_rate'] = available_trans_df['holding_days'].apply(
            lambda d: cc_stcg if d <= 365 else cc_ltcg
        )
        available_trans_df['tax'] = available_trans_df['gains'] * available_trans_df['applicable_tax_rate']
        available_trans_df['post_tax_current_value'] = available_trans_df['current_value'] - available_trans_df['tax']
        
        remaining_amount = amount
        trans_ids_to_remove = []
        trans_ids_to_update = {}
        total_units_withdrawn = 0
        total_pretax_amount = 0
        total_tax_paid = 0
        
        for id_, row_ in available_trans_df.iterrows():
            if remaining_amount <= 0: break
            
            available_val = row_['post_tax_current_value']
            
            if remaining_amount >= available_val:
                remaining_amount -= available_val
                trans_ids_to_remove.append(id_)
                total_units_withdrawn += row_['units']
                total_pretax_amount += row_['current_value']
                total_tax_paid += row_['tax']
            else:
                fraction = remaining_amount / available_val
                units_wd = row_['units'] * fraction
                pretax_wd = row_['current_value'] * fraction
                tax_wd = row_['tax'] * fraction
                
                total_units_withdrawn += units_wd
                total_pretax_amount += pretax_wd
                total_tax_paid += tax_wd
                
                trans_ids_to_update[id_] = {
                    'units': row_['units'] - units_wd,
                    'Amount': row_['Amount'] * (1 - fraction)
                }
                remaining_amount = 0
        
        fully_funded = (remaining_amount <= 1e-6)
        
        # Apply updates
        updated_trans_df = updated_trans_df.drop(trans_ids_to_remove)
        for id_, updates in trans_ids_to_update.items():
            updated_trans_df.loc[id_, 'units'] = updates['units']
            updated_trans_df.loc[id_, 'Amount'] = updates['Amount']
            
        updated_trans_df = updated_trans_df.reset_index(drop=True)
        
        if fully_funded:
            withdrawal_transactions.append({
                'Date': date, 'Amount': -total_pretax_amount, 'NAV': current_nav,
                'units': -total_units_withdrawn, 'Description': description,
                'tax': total_tax_paid, 'fully_funded': True, 'shortfall': 0
            })
        else:
            withdrawal_transactions.append({
                'Date': date, 'Amount': -amount, 'NAV': current_nav,
                'units': -amount/current_nav, 'Description': description,
                'tax': 0, 'fully_funded': False, 'shortfall': remaining_amount
            })
            
    # Combine
    sip_trans_final = sip_trans_df.copy()
    sip_trans_final['tax'] = 0
    sip_trans_final['fully_funded'] = True
    sip_trans_final['shortfall'] = 0
    
    trans_df = pd.concat([sip_trans_final, pd.DataFrame(withdrawal_transactions)], ignore_index=True)
    trans_df = trans_df.sort_values('Date').reset_index(drop=True)
    
    failed = trans_df[trans_df['fully_funded'] == False]
    success = len(failed) == 0
    
    failure_details = None
    if not success:
        first_fail = failed.iloc[0]
        failure_details = {
            'date': first_fail['Date'],
            'amount': abs(first_fail['shortfall']), # Shortfall amount
            'description': first_fail['Description']
        }
    
    return trans_df, success, failure_details

def get_default_glide_paths():
    excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Glide Paths.xlsx')
    return {
        'Non-Negotiable': pd.read_excel(excel_path, sheet_name='Non-Negotiable'),
        'Semi-Negotiable': pd.read_excel(excel_path, sheet_name='Semi-Negotiable'),
        'Negotiable': pd.read_excel(excel_path, sheet_name='Negotiable')
    }

def calculate_daily_value(final_trans_df, nav_df):
    trans_df = final_trans_df.copy(deep=True)
    trans_df['Date'] = trans_df['Date'].astype(_NS_DTYPE)
    trans_df = trans_df.sort_values('Date').reset_index(drop=True)

    trans_df = trans_df.groupby('Date', as_index=False)['units'].sum()

    trans_df['cumulative_units'] = trans_df['units'].cumsum()
    units_df = trans_df[['Date', 'cumulative_units']]

    units_df['Date'] = units_df['Date'].astype(_NS_DTYPE)
    units_df = units_df.sort_values('Date')
    
    if units_df.empty:
        return pd.DataFrame(columns=['Date', 'cumulative_units', 'nav', 'current_value'])
        
    full_dates = pd.date_range(
        start=units_df['Date'].min(),
        end=units_df['Date'].max(),
        freq='D'
    )

    units_df = (
        units_df
        .set_index('Date')
        .reindex(full_dates)
    )

    units_df['cumulative_units'] = units_df['cumulative_units'].ffill()
    units_df = units_df.reset_index().rename(columns={'index': 'Date'})

    units_df = units_df.merge(nav_df, on='Date', how='left')
    
    units_df['nav'] = units_df['nav'].ffill()
    units_df['current_value'] = units_df['cumulative_units'] * units_df['nav']

    return units_df

# --- Main Simulation Logic ---

def run_simulation(config, retirement_date, instrument_params, glide_paths=None):
    if glide_paths is None:
        glide_paths = get_default_glide_paths()

    current_date = config['current_date']
    target_lifetime = config.get('target_lifetime', 90)
    current_age = config.get('current_age', 30)
    death_date = pd.Timestamp(current_date + pd.DateOffset(years=int(target_lifetime - current_age)))

    # 1. Calculate Goal Cashflows
    goal_dfs = {}
    last_goal_date = current_date

    for goal in config['goals']:
        goal_end_date = goal['maturity_date']
        if goal_end_date > last_goal_date:
            last_goal_date = goal_end_date

        inflation = goal['rate_for_future_value%'] / 100
        fv_downpayment = future_value(goal['downpayment_present_value'], inflation, current_date, goal_end_date)

        goal_df = calculate_goal_cashflows(
            input_df=glide_paths[goal['type']],
            end_date=goal_end_date,
            goal_value_post_tax=fv_downpayment,
            instrument_params=instrument_params,
            input_variables=config
        )
        goal_dfs[goal['name']] = goal_df

    # 2. Setup Date Ranges
    # Simulation runs until death_date. Expenses and pools are funded exactly to death_date.
    final_date = max(last_goal_date, death_date)
    
    # 3. Generate NAV for Core Corpus (extended to 150 years)
    nav_df = generate_pseudo_nav(current_date, final_date, instrument_params['core_corpus']['return'])
    
    # 4. Calculate SIP Cashflows (Stop at Retirement)
    sip_df = calculate_sip_cashflows(config, final_date, retirement_date)
    
    # 5. Calculate Expense Cashflows (extended to 150 years for pool calculations)
    expense_df = calculate_expenses_cashflows(config, retirement_date, final_date)

    # 5a. Calculate Passive Income Cashflows (post-retirement only, grows from today's values)
    passive_income_df = calculate_passive_income_cashflows(config, retirement_date, final_date)

    # 6. Simulate Post-Retirement Pools (Debt & Hybrid)
    # This will generate "Replenishment Withdrawals" from Core Corpus
    # We generate NAVs from current_date because goals might use these pools before retirement.
    debt_nav_df = generate_pseudo_nav(current_date, final_date, instrument_params['debt']['return'])
    hybrid_nav_df = generate_pseudo_nav(current_date, final_date, instrument_params['hybrid']['return'])
    
    # We need to filter expenses to those after retirement
    post_ret_expense_df = expense_df[expense_df['Date'] >= retirement_date].copy().reset_index(drop=True)
    
    pool_trans_df, core_replenishments_df, post_ret_inflows_df, failure_date, failure_reason, expense_movements_df = simulate_post_retirement(
        post_ret_expense_df,
        debt_nav_df,
        hybrid_nav_df,
        instrument_params['debt'],
        instrument_params['hybrid'],
        retirement_date,
        final_date,
        income_df=passive_income_df
    )
    
    if failure_date:
        return False, pool_trans_df, {'date': failure_date, 'amount': 0, 'description': failure_reason}, expense_movements_df, {}, pd.DataFrame()

    # 7. Prepare Master Withdrawal List for Core Corpus
    # Goals
    withdrawals_from_goals = get_withdrawl_df(goal_dfs)
    
    # Core Replenishments (instead of direct expenses)
    # core_replenishments_df has columns: Date, Amount, Description
    
    all_withdrawals = pd.concat([withdrawals_from_goals, core_replenishments_df], ignore_index=True)

    # Negative net SIP amounts (effects exceeding SIP) become withdrawals from core corpus
    negative_sip = sip_df[sip_df['net sip amount'] < 0][['Date', 'net sip amount']].copy()
    if not negative_sip.empty:
        negative_sip['Amount'] = negative_sip['net sip amount'].abs()
        negative_sip['Description'] = 'Negative SIP Effect'
        all_withdrawals = pd.concat([all_withdrawals, negative_sip[['Date', 'Amount', 'Description']]], ignore_index=True)

    # 8. Run Transaction Simulation for Core Corpus
    sip_trans_df = create_sip_trans(nav_df, sip_df, config, retirement_date)

    # Add passive income surplus inflows back into core corpus
    if post_ret_inflows_df is not None and not post_ret_inflows_df.empty:
        inflow_rows = []
        for _, inflow_row in post_ret_inflows_df.iterrows():
            idate = inflow_row['Date']
            iamount = inflow_row['Amount']
            matches = nav_df[nav_df['Date'] <= idate]
            inav = matches['nav'].iloc[-1] if not matches.empty else nav_df['nav'].iloc[0]
            inflow_rows.append({
                'Date': idate, 'Amount': iamount, 'NAV': inav,
                'units': iamount / inav, 'Description': inflow_row['Description']
            })
        sip_trans_df = pd.concat([sip_trans_df, pd.DataFrame(inflow_rows)], ignore_index=True)
        sip_trans_df = sip_trans_df.sort_values('Date').reset_index(drop=True)

    final_trans_df, success, failure_details = add_withdrawls_to_trans(sip_trans_df, all_withdrawals, nav_df, instrument_params)
    
    # Merge pool transactions for complete record?
    # The user asked for "consolidated data frame" in previous turns, but run_simulation returns final_trans_df.
    # We should probably return the Core Corpus transactions primarily for success check.
    # But maybe append pool transactions? 
    # For now, let's keep final_trans_df as Core Corpus history, as that determines "Can Retire".
    
    # 9. Generate Comprehensive View
    comprehensive_df = generate_comprehensive_view(
        config, final_trans_df, pool_trans_df, goal_dfs,
        nav_df, debt_nav_df, hybrid_nav_df,
        sip_df, expense_df, passive_income_df
    )

    return success, final_trans_df, failure_details, expense_movements_df, goal_dfs, comprehensive_df

def generate_comprehensive_view(config, final_trans_df, pool_trans_df, goal_dfs, nav_df, debt_nav_df, hybrid_nav_df, sip_df, expense_df, passive_income_df=None):
    current_date = config['current_date']
    target_lifetime = config.get('target_lifetime', 90)
    current_age = config.get('current_age', 30)
    death_date = pd.Timestamp(current_date + pd.DateOffset(years=int(target_lifetime - current_age)))

    end_date = final_trans_df['Date'].max()
    if pool_trans_df is not None and not pool_trans_df.empty:
        end_date = max(end_date, pool_trans_df['Date'].max())
    # Always extend at least to death_date so the chart reaches the full target lifetime
    end_date = max(end_date, death_date)

    # Generate Month-End dates up to end_date
    full_date_range = pd.date_range(start=current_date, end=end_date, freq='ME')
    # 'M' is deprecated for Month End in recent pandas, 'ME' is better, or just use 'M' if on older. 
    # Let's use 'M' to be safe or 'D' and filter? 'M' is month end.
    
    # Create the master DF
    master_df = _ensure_date_ns(pd.DataFrame({'Date': full_date_range}))

    # 1. Core Corpus Value
    # Calculate daily units similar to calculate_daily_value but for just these dates?
    # Better: Calculate daily units series, then join.
    # We can reuse logic or just do it here efficiently.

    core_trans = final_trans_df.copy()
    core_trans['Date'] = core_trans['Date'].astype(_NS_DTYPE)
    core_trans = core_trans.sort_values('Date')
    core_daily_cats = _ensure_date_ns(pd.DataFrame({'Date': pd.date_range(start=current_date, end=end_date, freq='D')}))
    
    # Agg transactions by day
    agg_trans = core_trans.groupby('Date')['units'].sum().reset_index()
    core_vals = core_daily_cats.merge(agg_trans, on='Date', how='left').fillna(0)
    core_vals['cum_units'] = core_vals['units'].cumsum()
    
    # Get NAVs
    # nav_df might be sparse? No, generate_pseudo_nav is daily.
    core_vals = core_vals.merge(nav_df[['Date', 'nav']], on='Date', how='left').ffill()
    core_vals['Core Corpus Value'] = core_vals['cum_units'] * core_vals['nav']
    
    # Merge into Master (Asof nearest? or exact date matching?)
    # Month ends should match exactly if generated correctly, but dates might differ slightly if 'ME' gives last day.
    # merge_asof is safest.
    master_df = pd.merge_asof(master_df, core_vals[['Date', 'Core Corpus Value']], on='Date')
    
    # 2. Expense Debt & Hybrid Pools
    # If pool_trans_df is empty (pre-retirement or failure), populate 0
    if pool_trans_df is not None and not pool_trans_df.empty:
        # Separate by Pool
        # Ensure Date is datetime
        pool_trans_df['Date'] = pool_trans_df['Date'].astype(_NS_DTYPE)

        for pool_name, nav_source in [('Debt', debt_nav_df), ('Hybrid', hybrid_nav_df)]:
            p_trans = pool_trans_df[pool_trans_df['Pool'] == pool_name].copy()
            if p_trans.empty:
                master_df[f'Expense {pool_name} Pool Value'] = 0.0
                continue

            agg_p = p_trans.groupby('Date')['units'].sum().reset_index()
            daily_p = _ensure_date_ns(pd.DataFrame({'Date': pd.date_range(start=current_date, end=end_date, freq='D')}))
            daily_p = daily_p.merge(agg_p, on='Date', how='left').fillna(0)
            daily_p['cum_units'] = daily_p['units'].cumsum()
            
            daily_p = daily_p.merge(nav_source[['Date', 'nav']], on='Date', how='left').ffill()
            daily_p['val'] = daily_p['cum_units'] * daily_p['nav']
            
            master_df = pd.merge_asof(master_df, daily_p[['Date', 'val']], on='Date')
            master_df = master_df.rename(columns={'val': f'Expense {pool_name} Pool Value'})
    else:
        master_df['Expense Debt Pool Value'] = 0.0
        master_df['Expense Hybrid Pool Value'] = 0.0

    # 3. Goal Specific Pools
    # goal_dfs: dict of name -> df
    for goal_name, df in goal_dfs.items():
        # df has: inflow_date, inflow_amount, place, outflow_date
        # We want to track value in Debt and Hybrid.
        
        # Initialize columns
        master_df[f'{goal_name} Debt Value'] = 0.0
        master_df[f'{goal_name} Hybrid Value'] = 0.0
        
        for idx, row in df.iterrows():
            place = row['place'].lower()
            if place not in ['debt', 'hybrid']:
                continue
                
            start_d = row['inflow_date']
            end_d = row['outflow_date'] if pd.notna(row['outflow_date']) else end_date
            amount = row['inflow_amount']
            
            if start_d >= end_d: continue
            
            # Select NAV DF
            curr_nav_df = debt_nav_df if place == 'debt' else hybrid_nav_df
            
            # Get Start NAV
            # Using simple lookup
            s_nav_rows = curr_nav_df[curr_nav_df['Date'] <= start_d]
            if s_nav_rows.empty: s_nav = curr_nav_df['nav'].iloc[0]
            else: s_nav = s_nav_rows['nav'].iloc[-1]
            
            units = amount / s_nav
            
            # Calculate value for the range
            # We filter master_df for relevant rows
            mask = (master_df['Date'] >= start_d) & (master_df['Date'] <= end_d)
            subset_dates = master_df.loc[mask, 'Date']
            
            # Get NAVs for these dates
            # merge
            temp_df = _ensure_date_ns(pd.DataFrame({'Date': subset_dates}))
            temp_df = pd.merge_asof(temp_df, curr_nav_df, on='Date')
            
            # Add to master
            # We map back by index or Date
            values = temp_df['nav'] * units
            
            col_name = f'{goal_name} {place.capitalize()} Value'
            master_df.loc[mask, col_name] += values.values

    # 4. SIP and Expenses (Net)
    # sip_df, expense_df are monthly.
    # sip_df: Date, net sip amount
    # expense_df: Date, Net Expense Amount
    
    # Merge SIP
    # Agg SIP by month (Date is Month Start usually)
    # Our Master DF is Month End.
    # We can match on Year-Month.
    
    master_df['YearMonth'] = master_df['Date'].dt.to_period('M')
    
    sip_agg = sip_df.copy()
    sip_agg['YearMonth'] = sip_agg['Date'].dt.to_period('M')
    sip_agg = sip_agg.groupby('YearMonth')['net sip amount'].sum().reset_index()
    
    master_df = master_df.merge(sip_agg, on='YearMonth', how='left').fillna({'net sip amount': 0})
    master_df = master_df.rename(columns={'net sip amount': 'Net SIP'})
    
    # Merge Expenses
    exp_agg = expense_df.copy()
    exp_agg['YearMonth'] = exp_agg['Date'].dt.to_period('M')
    exp_agg = exp_agg.groupby('YearMonth')['Net Expense Amount'].sum().reset_index()
    
    master_df = master_df.merge(exp_agg, on='YearMonth', how='left').fillna({'Net Expense Amount': 0})
    master_df = master_df.rename(columns={'Net Expense Amount': 'Net Expenses'})
    
    # Merge Passive Income
    if passive_income_df is not None and not passive_income_df.empty:
        income_agg = passive_income_df.copy()
        income_agg['YearMonth'] = income_agg['Date'].dt.to_period('M')
        income_agg = income_agg.groupby('YearMonth')['Passive Income Amount'].sum().reset_index()
        master_df = master_df.merge(income_agg, on='YearMonth', how='left').fillna({'Passive Income Amount': 0})
        master_df = master_df.rename(columns={'Passive Income Amount': 'Passive Income'})
    else:
        master_df['Passive Income'] = 0.0

    # Cleanup
    master_df = master_df.drop(columns=['YearMonth'])

    return master_df

def calculate_debt_injection_need(expenses_list, injection_date, pool_params):
    # expenses_list: list of (date, amount)
    # Returns PV needed at injection_date to meet these expenses

    total_pv = 0
    rate = pool_params['return']
    stcg_tax = pool_params['stcg_tax']
    ltcg_tax = pool_params['ltcg_tax']

    for date, amount in expenses_list:
        if date < injection_date: continue
        years_to_expense = (date - injection_date).days / 365.25
        # Prevent negative years
        years_to_expense = max(0, years_to_expense)

        tax_rate = stcg_tax if years_to_expense <= 1 else ltcg_tax
        needed = calculate_corpus_required_for_future_expense(amount, years_to_expense, rate, tax_rate)
        total_pv += needed

    return total_pv

def simulate_post_retirement(expense_df, debt_nav_df, hybrid_nav_df, debt_params, hybrid_params, retirement_date, final_date, income_df=None):
    # Returns: pool_trans_df, core_replenishments_df, post_ret_inflows_df, failure_date, failure_reason, expense_movements_df

    debt_pool = InvestmentPool('Debt', debt_params['stcg_tax'], debt_params['ltcg_tax'])
    hybrid_pool = InvestmentPool('Hybrid', hybrid_params['stcg_tax'], hybrid_params['ltcg_tax'])

    pool_transactions = []
    core_replenishments = []
    post_ret_inflows = []
    expense_movements = []

    # Build monthly income lookup {(year, month): amount} for O(1) access
    income_data_monthly = {}
    if income_df is not None and not income_df.empty:
        for _, row in income_df.iterrows():
            key = (row['Date'].year, row['Date'].month)
            income_data_monthly[key] = income_data_monthly.get(key, 0) + row['Passive Income Amount']

    # Create quick access to NAVs
    # Using asof merge later might be cleaner, but for loop lookups are ok if optimized
    # Convert to dict for O(1) mostly? Date range is small (100 years = 1200 points).
    debt_nav_dict = dict(zip(debt_nav_df['Date'], debt_nav_df['nav']))
    hybrid_nav_dict = dict(zip(hybrid_nav_df['Date'], hybrid_nav_df['nav']))
    
    def get_nav(date, nav_dict, default_df):
        if date in nav_dict: return nav_dict[date]
        # Fallback to nearest
        matches = default_df[default_df['Date'] <= date]
        if not matches.empty: return matches['nav'].iloc[-1]
        return default_df['nav'].iloc[-1]
        
    def log_movement(date, debt_in=0, debt_out=0, hybrid_in=0, hybrid_out=0):
        d_nav = get_nav(date, debt_nav_dict, debt_nav_df)
        h_nav = get_nav(date, hybrid_nav_dict, hybrid_nav_df)
        
        d_val = debt_pool.get_market_value(d_nav)
        h_val = hybrid_pool.get_market_value(h_nav)
        
        expense_movements.append({
            'Date': date,
            'Debt Pool Value': d_val,
            'Inflow to Debt': debt_in,
            'Outflow from Debt': debt_out,
            'Hybrid Pool Value': h_val,
            'Inflow to Hybrid': hybrid_in,
            'Outflow from Hybrid': hybrid_out
        })

    sim_date = retirement_date
    
    # Pre-process expenses into a dict or sorted list for range queries
    expense_data = list(zip(expense_df['Date'], expense_df['Net Expense Amount']))
    
    while sim_date <= final_date:
        # 1. Start of Year (Replenishment Cycle)
        # Note: Replenishment happens every 12 months.
        # Check if expenses exist beyond this point. If not, break.
        if sim_date > expense_df['Date'].max():
            break
            
        debt_nav = get_nav(sim_date, debt_nav_dict, debt_nav_df)
        hybrid_nav = get_nav(sim_date, hybrid_nav_dict, hybrid_nav_df)
        
        # --- A. Determine Needs ---
        # Debt Window: Next 24 months — use net expenses (expense minus passive income)
        debt_deadline = sim_date + pd.DateOffset(months=24)
        debt_net_expenses = [(d, max(0, a - income_data_monthly.get((d.year, d.month), 0))) for d, a in expense_data if sim_date <= d < debt_deadline]

        target_debt_val = calculate_debt_injection_need(debt_net_expenses, sim_date, debt_params)

        # Hybrid Window: Months 25–60 — same net expense logic
        hybrid_window_start = sim_date + pd.DateOffset(months=24)
        hybrid_window_end = sim_date + pd.DateOffset(months=60)

        hybrid_net_expenses = [(d, max(0, a - income_data_monthly.get((d.year, d.month), 0))) for d, a in expense_data if hybrid_window_start <= d < hybrid_window_end]

        target_hybrid_val = calculate_debt_injection_need(hybrid_net_expenses, sim_date, hybrid_params)

        # --- B. Execute Transfers ---
        # 1. Check Hybrid Surplus/Shortfall
        current_hybrid_val = hybrid_pool.get_market_value(hybrid_nav)
        hybrid_latent_tax = hybrid_pool.get_unrealized_tax(hybrid_nav, sim_date)
        # Adjusted Target = Target(Fresh) + Latent Tax
        # Shortfall = (Target + Tax) - CurrentVal
        # Surplus = CurrentVal - (Target + Tax)
        hybrid_shortfall = max(0, (target_hybrid_val + hybrid_latent_tax) - current_hybrid_val)
        hybrid_surplus = max(0, current_hybrid_val - (target_hybrid_val + hybrid_latent_tax))

        # 2. Check Debt Shortfall
        current_debt_val = debt_pool.get_market_value(debt_nav)
        debt_latent_tax = debt_pool.get_unrealized_tax(debt_nav, sim_date)
        debt_shortfall = max(0, (target_debt_val + debt_latent_tax) - current_debt_val)
        
        # 3. Hybrid -> Debt (Surplus only)
        if debt_shortfall > 0 and hybrid_surplus > 0:
            transfer_gross = min(hybrid_surplus, debt_shortfall) # Using gross value transfer as approximation of "Moving Surplus"
            
            # Redeem from Hybrid
            # Redeem from Hybrid
            wd_res = hybrid_pool.redeem_gross_amount(sim_date, transfer_gross, hybrid_nav, description="Transfer to Debt (Surplus)")
            pool_transactions.append(wd_res)

            
            net_proceeds = wd_res['net_received']
            
            # Invest in Debt
            inv_res = debt_pool.invest(sim_date, net_proceeds, debt_nav, description="Transfer from Hybrid")
            if inv_res: pool_transactions.append(inv_res)
            
            # Log Hybrid -> Debt
            log_movement(sim_date, debt_in=net_proceeds, hybrid_out=transfer_gross) # Gross out from hybrid, net in to debt
            
            # Recalculate Debt Shortfall
            current_debt_val = debt_pool.get_market_value(debt_nav)
            debt_latent_tax = debt_pool.get_unrealized_tax(debt_nav, sim_date)
            debt_shortfall = max(0, (target_debt_val + debt_latent_tax) - current_debt_val)

        # 4. Core -> Debt (Remaining Shortfall)
        if debt_shortfall > 0.01: # Check > 0 with tolerance
            # We request exactly debt_shortfall. 
            # Note: debt_shortfall is "Value Gap". If I add Cash $X, Value increases by $X immediately? Yes.
            core_replenishments.append({'Date': sim_date, 'Amount': debt_shortfall, 'Description': 'Replenishment: Debt Pool'})
            inv_res = debt_pool.invest(sim_date, debt_shortfall, debt_nav, description="Replenishment from Core")
            if inv_res: pool_transactions.append(inv_res)
            
            log_movement(sim_date, debt_in=debt_shortfall)


        # 5. Core -> Hybrid (Refill to Target)
        current_hybrid_val = hybrid_pool.get_market_value(hybrid_nav)
        hybrid_latent_tax = hybrid_pool.get_unrealized_tax(hybrid_nav, sim_date)
        hybrid_shortfall = max(0, (target_hybrid_val + hybrid_latent_tax) - current_hybrid_val)
        
        if hybrid_shortfall > 0.01:
            core_replenishments.append({'Date': sim_date, 'Amount': hybrid_shortfall, 'Description': 'Replenishment: Hybrid Pool'})
            inv_res = hybrid_pool.invest(sim_date, hybrid_shortfall, hybrid_nav, description="Replenishment from Core")
            if inv_res: pool_transactions.append(inv_res)
            
            log_movement(sim_date, hybrid_in=hybrid_shortfall)

            
        # --- C. Monthly Withdrawals (Expense Loop for next 12 months) ---
        next_year = sim_date + pd.DateOffset(months=12)
        
        # Loop months
        m_date = sim_date
        while m_date < next_year:
            # Check expenses in this month
            # (Assuming one expense per month for simplicity as per struct)
            # Find expense entry
            month_expenses = [a for d, a in expense_data if d.year == m_date.year and d.month == m_date.month]
            
            if not month_expenses:
                m_date += relativedelta(months=1)
                continue
                
            total_expense = sum(month_expenses)
            month_income = income_data_monthly.get((m_date.year, m_date.month), 0)
            net_withdrawal = max(0, total_expense - month_income)
            surplus = max(0, month_income - total_expense)

            if net_withdrawal > 0:
                curr_nav = get_nav(m_date, debt_nav_dict, debt_nav_df)
                wd_res = debt_pool.redeem_net_amount(m_date, net_withdrawal, curr_nav, description="Monthly Expense")
                pool_transactions.append(wd_res)
                log_movement(m_date, debt_out=net_withdrawal)
                if not wd_res['fully_funded']:
                    return pd.DataFrame(pool_transactions), pd.DataFrame(core_replenishments), pd.DataFrame(post_ret_inflows), m_date, "Debt Pool Depleted", pd.DataFrame(expense_movements)
            else:
                if surplus > 0:
                    post_ret_inflows.append({'Date': m_date, 'Amount': surplus, 'Description': 'Passive Income Surplus'})
                log_movement(m_date)

            
            m_date += relativedelta(months=1)
            
        sim_date = next_year

    return pd.DataFrame(pool_transactions), pd.DataFrame(core_replenishments), pd.DataFrame(post_ret_inflows), None, None, pd.DataFrame(expense_movements)



def find_retirement_date(config, instrument_params=None, glide_paths=None):
    current_date = config['current_date']
    target_lifetime = config.get('target_lifetime', 90)
    current_age = config.get('current_age', 30)
    death_date = pd.Timestamp(current_date + pd.DateOffset(years=int(target_lifetime - current_age)))

    if instrument_params is None:
        instrument_params = {
            'core_corpus': {'return': 0.12, 'stcg_tax': 0.20, 'ltcg_tax': 0.125},
            'equity': {'return': 0.12, 'stcg_tax': 0.20, 'ltcg_tax': 0.125},
            'debt': {'return': 0.08, 'stcg_tax': 0.20, 'ltcg_tax': 0.125},
            'hybrid': {'return': 0.12, 'stcg_tax': 0.20, 'ltcg_tax': 0.125},
            'cash': {'return': 0.04, 'stcg_tax': 0.20, 'ltcg_tax': 0.125}
        }

    if glide_paths is None:
        glide_paths = get_default_glide_paths()

    start_months = current_date.year * 12 + current_date.month
    end_months = death_date.year * 12 + death_date.month
    
    low = start_months
    high = end_months
    result = None
    
    # Optimization: If we can retire at `low`, return it.
    # We want EARLIEST date. Binary search might not work if function is not monotonic?
    # Generally, delaying retirement adds more corpus and reduces years of expense, so it should be monotonic (easier to retire later).
    # So binary search is valid. we want smallest `mid` that returns True.
    
    while low <= high:
        mid = (low + high) // 2
        year = mid // 12
        month = mid % 12
        if month == 0:
            month = 12
            year -= 1
            
        retirement_date = pd.Timestamp(year=year, month=month, day=1)
        
        # Run simulation with pre-loaded stuff
        success, _, _, _, _, _ = run_simulation(config, retirement_date, instrument_params, glide_paths)

        
        if success:
            result = (month, year)
            high = mid - 1 # Try earlier
        else:
            low = mid + 1 # Need more time
            
    return result

def main():
    config = {
        "current_date": pd.Timestamp("2026-01-09 00:00:00"),
        "current_age": 30,
        "target_lifetime": 90,
        "current_corpus": 10000000,
        "current_sip": 100000,
        "yearly_sip_step_up_%": 10.0,
        "stepup_date_month": None,
        "stepup_date_day": None,
        "sip_adjustments": [],
        "expense_streams": [
            {
                "name": "Living Expenses",
                "amount": 50000,
                "frequency": 1,
                "inflation": 6.0,
                "post_retirement_change": -15.0,
                "adjustments": []
            }
        ],
        "goals": [
            {
                "name": "Retirement Home",
                "type": "Non-Negotiable",
                "maturity_date": pd.Timestamp("2040-01-01 00:00:00"),
                "downpayment_present_value": 5000000,
                "rate_for_future_value%": 6.0
            }
        ],
        "effects_on_cashflows": []
    }

    print("Finding Retirement Date...")
    result = find_retirement_date(config)
    if result:
        print(f"Retirement Possible at: {result[0]}/{result[1]}")
    else:
        print("Cannot retire within 100 years.")

if __name__ == "__main__":
    main()
