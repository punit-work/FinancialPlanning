import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def calculate_goal_cashflows(input_df, end_date, goal_value_post_tax, instrument_params, input_variables):

    current_date = input_variables['current_date']
    
    # Create working copy
    df = input_df.copy()
    
    # Convert end_date to pd.Timestamp if not already
    end_date = pd.Timestamp(end_date)
    
    # Normalize place names to lowercase for consistency
    df['place'] = df['place'].str.lower()
    
    # Calculate inflow_date using relativedelta
    df['inflow_date'] = df['years from inflow till end'].apply(
        lambda years: end_date - relativedelta(years=int(years))
    )
    
    # Calculate outflow_date using relativedelta
    # If years from outflow till end is 0, outflow happens at end_date
    # If NaN, there is no outflow (goal rows)
    df['outflow_date'] = df['years from outflow till end'].apply(
        lambda x: end_date - relativedelta(years=int(x)) if pd.notna(x) else pd.NaT
    )

    # Ensure columns are datetime
    df['inflow_date'] = pd.to_datetime(df['inflow_date'])
    df['outflow_date'] = pd.to_datetime(df['outflow_date'])

    # Replace dates earlier than current_date with current_date
    df[['inflow_date', 'outflow_date']] = df[['inflow_date', 'outflow_date']].mask(
        df[['inflow_date', 'outflow_date']] < current_date,
        current_date
    )
    
    # Add goal_value_post_tax column
    df['goal_value_post_tax'] = goal_value_post_tax
    
    # Initialize inflow_amount column
    df['inflow_amount'] = 0.0
    
    # Create id to index mapping
    id_to_idx = {row['id']: idx for idx, row in df.iterrows()}
    
    # Helper function to calculate required inflow
    def calculate_required_inflow(target_post_tax, annual_return, tax_rate, years):
        if years == 0:
            return target_post_tax
        growth_factor = (1 + annual_return) ** years
        multiplier = growth_factor * (1 - tax_rate) + tax_rate
        return target_post_tax / multiplier
    
    # Helper function to process a single cashflow chain
    def process_chain(goal_row_id):
        current_id = goal_row_id
        current_idx = id_to_idx[current_id]
        current_row = df.loc[current_idx]
        
        # Set target amount for goal
        target_amount = current_row['goal_value_post_tax'] * (current_row['% of goal value'] / 100)
        df.at[current_idx, 'inflow_amount'] = target_amount
        
        # Work backwards through the chain
        while True:
            current_idx = id_to_idx[current_id]
            current_row = df.loc[current_idx]
            inflow_from = current_row['inflow_from']
            
            # Stop if we've reached core corpus
            if inflow_from == 'core corpus':
                break
            
            # Get source row
            source_idx = id_to_idx[inflow_from]
            source_row = df.loc[source_idx]
            
            # Calculate investment period in years
            inflow_date = source_row['inflow_date']
            outflow_date = current_row['inflow_date']
            years = (outflow_date - inflow_date).days / 365.25
            
            # Get instrument parameters
            place = source_row['place'].lower()
            params = instrument_params.get(place, {'return': 0.0, 'tax': 0.0})
            annual_return = params['return']
            tax_rate = params['tax']
            
            # Calculate required inflow
            target_for_source = df.at[current_idx, 'inflow_amount']
            required_inflow = calculate_required_inflow(
                target_for_source,
                annual_return,
                tax_rate,
                years
            )
            
            df.at[source_idx, 'inflow_amount'] = required_inflow
            current_id = inflow_from
    
    # Find all goal rows and process each chain
    goal_rows = df[df['place'] == 'goal']
    for _, goal_row in goal_rows.iterrows():
        process_chain(goal_row['id'])
    
    # Round inflow amounts
    df['inflow_amount'] = df['inflow_amount'].round(2)
    
    # Calculate outflow amounts and taxes
    df['total_outflow_amount'] = 0.0
    df['tax_out_of_outflow'] = 0.0
    
    for idx, row in df.iterrows():
        # Skip goal rows - they don't have outflows
        if row['place'] == 'goal':
            df.at[idx, 'total_outflow_amount'] = pd.NA
            df.at[idx, 'tax_out_of_outflow'] = pd.NA
            continue
        
        # Get instrument parameters
        place = row['place'].lower()
        params = instrument_params.get(place, {'return': 0.0, 'tax': 0.0})
        annual_return = params['return']
        tax_rate = params['tax']
        
        # Calculate investment period in years
        if pd.notna(row['outflow_date']):
            years = (row['outflow_date'] - row['inflow_date']).days / 365.25
            
            # Calculate total amount after growth
            principal = row['inflow_amount']
            total_outflow = principal * ((1 + annual_return) ** years)
            
            # Calculate gains and tax
            gains = total_outflow - principal
            tax = gains * tax_rate
            
            df.at[idx, 'total_outflow_amount'] = round(total_outflow, 2)
            df.at[idx, 'tax_out_of_outflow'] = round(tax, 2)
        else:
            df.at[idx, 'total_outflow_amount'] = pd.NA
            df.at[idx, 'tax_out_of_outflow'] = pd.NA
    
    # Select and order output columns
    output_columns = [
        'id', 'place', 'inflow_date', 'outflow_date', 'inflow_from',
        'outflow_to', '% of goal value', 'goal_value_post_tax', 'inflow_amount',
        'total_outflow_amount', 'tax_out_of_outflow'
    ]
    
    return df[output_columns]

def future_value(present_value, inflation_rate, current_date, future_date):

    # Time difference in years (actual days / 365.25)
    years = (future_date - current_date).days / 365.25

    # Future value calculation
    fv = present_value * ((1 + inflation_rate) ** years)

    return round(fv, 2)

def calculate_sip_cashflows(input_variables, last_goal_date):
    # Extract variables
    current_date = input_variables['current_date']
    current_sip = input_variables['current_sip']
    yearly_step_up = input_variables['yearly_sip_step_up_%'] / 100
    goals = input_variables['goals']
    effects_on_cashflows = input_variables['effects_on_cashflows']
    
    # Find the last maturity date
    last_maturity_date = last_goal_date
    
    # Create date range from current_date to last maturity date
    date_range = pd.date_range(start=current_date, end=last_maturity_date, freq='MS')
    
    # Initialize DataFrame
    df = pd.DataFrame({'Date': date_range})
    
    # Calculate SIP amounts with yearly step-up
    sip_amounts = []
    for i, date in enumerate(date_range):
        # Calculate years elapsed from start
        months_elapsed = i
        years_elapsed = months_elapsed // 12
        
        # Apply step-up formula: current_sip * (1 + step_up)^years_elapsed
        sip_amount = current_sip * ((1 + yearly_step_up) ** years_elapsed)
        sip_amounts.append(sip_amount)
    
    df['SIP amount'] = sip_amounts
    
    # Calculate adjustment amounts based on effects_on_cashflows
    df['adjustment amount'] = 0.0
    
    for effect in effects_on_cashflows:
        start_date = effect['start_date']
        end_date = effect['end_date']
        monthly_amount = effect['monthly_amount']
        
        # Apply adjustment for dates within the range
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        df.loc[mask, 'adjustment amount'] += monthly_amount
    
    # Calculate net cashflow amount
    df['net sip amount'] = df['SIP amount'] + df['adjustment amount']
    
    return df

def generate_pseudo_nav(start_date, end_date, rate_of_return):
    start_date = pd.Timestamp('2000-01-01')
    end_date = pd.Timestamp('2200-01-01')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    annual_rate = rate_of_return
    daily_rate = (1 + annual_rate) ** (1/365) - 1
    
    days_elapsed = np.arange(len(date_range))
    nav_values = 100 * (1 + daily_rate) ** days_elapsed
    
    pseudo_nav_df = pd.DataFrame({
        'Date': date_range,
        'nav': nav_values
    })
    
    return pseudo_nav_df

def create_sip_trans(nav_df, sip_df, input_variables):
    trans = []

    current_corpus = input_variables['current_corpus']
    current_date = input_variables['current_date']
    nav = nav_df[nav_df['Date']==current_date]['nav'].iloc[-1]
    units = current_corpus / nav

    trans.append({
        'Date': input_variables['current_date'],
        'Amount': input_variables['current_corpus'],
        'NAV': nav,
        'units': units,
        'Description': 'Current Corpus'
    })

    for id, row in sip_df.iterrows():
        amount = row['net sip amount']
        date = row['Date']
        nav = nav_df[nav_df['Date']<=date]['nav'].iloc[-1]
        units = amount / nav

        trans.append({
            'Date': date,
            'Amount': amount,
            'NAV': nav,
            'units': units,
            'Description': 'Monthly inflow'
        })

    # for name, goal_df in goal_dfs.items():
    #     for g_id, g_row in goal_df.iterrows():
    #         if g_row['inflow_from'] == 'core corpus':
    #             amount = g_row['inflow_amount']
    #             date = g_row['inflow_date']
    #             nav = nav_df[nav_df['Date']<=date]['nav'].iloc[-1]
    #             units = amount / nav

    #             trans.append({
    #                 'Date': date,
    #                 'Amount': -amount,
    #                 'NAV': nav,
    #                 'units': -units,
    #                 'Description': f'For Goal - {name}'
    #             })

    return pd.DataFrame(trans)

def get_withdrawl_df(goal_dfs):

    results = []

    for name, df in goal_dfs.items():
        for id, row in df[df['inflow_from']=='core corpus'].copy(deep=True).sort_values(by='inflow_date').reset_index(drop=True).iterrows():
            date = row['inflow_date']
            amount = row['inflow_amount']
            place = row['place']
            description = f'Moving to {place} for {name} goal.'
            results.append({
                'Date': date,
                'Amount': amount,
                'Description': description
            })

    return pd.DataFrame(results)

def add_withdrawls_to_trans(sip_trans_df, withdrawls_df, nav_df, instrument_params):
    updated_trans_df = sip_trans_df.copy(deep=True)
    withdrawal_transactions = []
    
    for id, row in withdrawls_df.iterrows():
        amount = row['Amount']  # Post-tax amount needed
        date = row['Date']
        description = row['Description']
        # st.dataframe(nav_df[nav_df['Date']==date]['nav'])
        current_nav = nav_df[nav_df['Date']==date]['nav'].iloc[-1]
        
        # Filter to only include transactions up to the withdrawal date
        available_trans_df = updated_trans_df[updated_trans_df['Date'] <= date].copy()
        
        if available_trans_df.empty:
            # No transactions available before this withdrawal date
            # Add transaction for the requested amount anyway
            units_needed = amount / current_nav
            withdrawal_transactions.append({
                'Date': date,
                'Amount': -amount,
                'NAV': current_nav,
                'units': -units_needed,
                'Description': description,
                'tax': 0,
                'fully_funded': False,
                'shortfall': amount
            })
            continue
        
        # Calculate current values and tax for all existing positions
        available_trans_df['current_value'] = available_trans_df['units'] * current_nav
        available_trans_df['gains'] = available_trans_df['current_value'] - available_trans_df['Amount']
        available_trans_df['tax'] = available_trans_df['gains'] * instrument_params['core_corpus']['tax']
        available_trans_df['post_tax_current_value'] = available_trans_df['current_value'] - available_trans_df['tax']
        
        remaining_amount = amount
        trans_ids_to_remove = []
        trans_ids_to_update = {}
        total_units_withdrawn = 0
        total_pretax_amount = 0
        total_tax_paid = 0
        
        for id_, row_ in available_trans_df.iterrows():
            if remaining_amount <= 0:
                break
                
            available_amount = row_['post_tax_current_value']
            
            if remaining_amount >= available_amount:
                # Withdraw entire position
                remaining_amount -= available_amount
                trans_ids_to_remove.append(id_)
                total_units_withdrawn += row_['units']
                total_pretax_amount += row_['current_value']
                total_tax_paid += row_['tax']
            else:
                # Partially withdraw from this position
                fraction = remaining_amount / available_amount
                units_to_withdraw = row_['units'] * fraction
                pretax_amount = row_['current_value'] * fraction
                tax_on_withdrawal = row_['tax'] * fraction
                
                total_units_withdrawn += units_to_withdraw
                total_pretax_amount += pretax_amount
                total_tax_paid += tax_on_withdrawal
                
                # Store update info
                trans_ids_to_update[id_] = {
                    'units': row_['units'] - units_to_withdraw,
                    'Amount': row_['Amount'] * (1 - fraction)
                }
                
                remaining_amount = 0
        
        # Check if withdrawal was fully funded
        withdrawal_fully_funded = (remaining_amount <= 1e-6)  # Small epsilon for floating point comparison
        
        # Update the main updated_trans_df with changes
        for id_ in trans_ids_to_remove:
            updated_trans_df = updated_trans_df.drop(id_)
        
        for id_, updates in trans_ids_to_update.items():
            updated_trans_df.loc[id_, 'units'] = updates['units']
            updated_trans_df.loc[id_, 'Amount'] = updates['Amount']
        
        updated_trans_df = updated_trans_df.reset_index(drop=True)
        
        # Add withdrawal transaction
        if withdrawal_fully_funded:
            # Normal case: withdrawal was fully funded
            withdrawal_transactions.append({
                'Date': date,
                'Amount': -total_pretax_amount,
                'NAV': current_nav,
                'units': -total_units_withdrawn,
                'Description': description,
                'tax': total_tax_paid,
                'fully_funded': True,
                'shortfall': 0
            })
        else:
            # Shortfall case: add transaction for the requested post-tax amount
            units_needed = amount / current_nav
            withdrawal_transactions.append({
                'Date': date,
                'Amount': -amount,  # Post-tax amount requested
                'NAV': current_nav,
                'units': -units_needed,
                'Description': description,
                'tax': 0,  # Can't calculate proper tax since funds weren't available
                'fully_funded': False,
                'shortfall': remaining_amount
            })
    
    # Add tax column and simulation tracking columns to original SIP transactions
    sip_trans_with_tax = sip_trans_df.copy()
    sip_trans_with_tax['tax'] = 0
    sip_trans_with_tax['fully_funded'] = True
    sip_trans_with_tax['shortfall'] = 0
    
    # Combine all transactions and sort by date
    trans_df = pd.concat([
        sip_trans_with_tax,
        pd.DataFrame(withdrawal_transactions)
    ], ignore_index=True).sort_values(by='Date').reset_index(drop=True)
    
    # Determine simulation status
    failed_withdrawals = trans_df[trans_df['fully_funded'] == False]
    
    if len(failed_withdrawals) > 0:
        last_successful_mask = trans_df['fully_funded'] == True
        if last_successful_mask.any():
            last_successful_date = trans_df[last_successful_mask]['Date'].iloc[-1]
        else:
            last_successful_date = None
        simulation_broke_date = failed_withdrawals.iloc[0]['Date']
        simulation_successful = False
        total_shortfall = failed_withdrawals['shortfall'].sum()
    else:
        last_successful_date = trans_df.iloc[-1]['Date']
        simulation_broke_date = None
        simulation_successful = True
        total_shortfall = 0
    
    return trans_df, {
        'simulation_successful': simulation_successful,
        'last_successful_date': last_successful_date,
        'simulation_broke_date': simulation_broke_date,
        'total_shortfall': total_shortfall,
        'num_failed_withdrawals': len(failed_withdrawals)
    }

def calculate_daily_value(final_trans_df, nav_df):

    trans_df = final_trans_df.copy(deep=True)
    trans_df['Date'] = pd.to_datetime(trans_df['Date'])
    trans_df = trans_df.sort_values('Date').reset_index(drop=True)

    trans_df = trans_df.groupby('Date', as_index=False)['units'].sum()

    trans_df['cumulative_units'] = trans_df['units'].cumsum()
    units_df = trans_df[['Date', 'cumulative_units']]

    units_df['Date'] = pd.to_datetime(units_df['Date'])
    units_df = units_df.sort_values('Date')
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

    units_df['current_value'] = units_df['cumulative_units'] * units_df['nav']

    return units_df

def format_inr(amount):
    amount = round(float(amount), 2)
    integer, decimal = f"{amount:.2f}".split(".")

    if len(integer) > 3:
        last3 = integer[-3:]
        rest = integer[:-3]
        rest = ",".join([rest[max(i-2, 0):i] for i in range(len(rest), 0, -2)][::-1])
        integer = rest + "," + last3

    return f"₹{integer}.{decimal}"


def run_simulation(input_variables, instrument_params):

    if not input_variables == None:

        glide_paths = {
            'Non-Negotiable': pd.read_excel('Glide Paths.xlsx', sheet_name='Non-Negotiable'),
            'Semi-Negotiable': pd.read_excel('Glide Paths.xlsx', sheet_name='Semi-Negotiable'),
            'Negotiable': pd.read_excel('Glide Paths.xlsx', sheet_name='Negotiable')
        }


        goal_dfs = {}
        last_goal_date = None
        current_date = input_variables['current_date']
        for goal in input_variables['goals']:
            goal_end_date = goal['maturity_date']
            goal_down_payment_current_value = goal['downpayment_present_value']
            inflation_rate = goal['rate_for_future_value%']/100
            goal_downpayment_future_value = future_value(goal_down_payment_current_value, inflation_rate, current_date, goal_end_date)
            type = goal['type']
            glide_path_df = glide_paths[type]

            goal_df = calculate_goal_cashflows(
                input_df=glide_path_df,
                end_date=goal_end_date,
                goal_value_post_tax=goal_downpayment_future_value,
                instrument_params=instrument_params,
                input_variables=input_variables
            )

            if last_goal_date == None or last_goal_date < goal_end_date:
                last_goal_date = goal_end_date

                goal_dfs[goal['name']] = goal_df.copy(deep=True)

        if last_goal_date == None:
            last_goal_date = input_variables['current_date'] + relativedelta(years=50)
        nav_df = generate_pseudo_nav(input_variables['current_date'], last_goal_date, instrument_params['core_corpus']['return'])
        
        sip_df = calculate_sip_cashflows(input_variables, last_goal_date)
        sip_trans_df = create_sip_trans(nav_df, sip_df, input_variables)
        withdrawls_df = get_withdrawl_df(goal_dfs)
        final_trans_df, success_metrics = add_withdrawls_to_trans(sip_trans_df, withdrawls_df, nav_df, instrument_params)

        daily_corpus_value_df = calculate_daily_value(final_trans_df, nav_df)

        # nav_df.to_excel('test/nav_df.xlsx', index=False)
        # sip_df.to_excel('test/sip_df.xlsx', index=False)
        # sip_trans_df.to_excel('test/sip_trans_df.xlsx', index=False)
        # withdrawls_df.to_excel('test/withdrawls_df.xlsx', index=False)
        # final_trans_df.to_excel('test/final_trans_df.xlsx', index=False)
        # daily_corpus_value_df.to_excel('test/daily_corpus_value_df.xlsx', index=False)

        # for name, df in goal_dfs.items():
        #     df.to_excel(f'test/{name}_goal_df.xlsx', index=False)

        if sip_df['net sip amount'].min() < 0:
            return {
                'status': 'error',
                'message': 'Monthly inflows become negative',
                'data': {
                    'sip_df': sip_df
                }
            }
        
        else:
            return {
                'status': 'success' if success_metrics['simulation_successful'] else 'failure',
                'data': {
                    'daily_corpus_value_df': daily_corpus_value_df,
                    'final_trans_df': final_trans_df,
                    'goal_dfs': goal_dfs,
                    'success_metrics': success_metrics,
                    'last_goal_date': last_goal_date
                }
            }
    return None

if __name__ == '__main__':
    # For testing purposes, you would need to mock input_variables and instrument_params here
    pass
