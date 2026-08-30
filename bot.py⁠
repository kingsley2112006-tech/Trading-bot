pip install python-binance pandas numpy
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

# Initialize Binance Client (Keys are not required for public historical data)
client = Client(api_key='', api_secret='')

# --- Configuration ---
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR  # 1-hour candles
LOOKBACK_DAYS = 30                      # Days of data to train the model

def get_historical_data(symbol, interval, lookback_days):
    """Fetches historical OHLCV data from Binance."""
    print(f"Fetching {lookback_days} days of historical data for {symbol}...")
    start_str = str((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
    klines = client.get_historical_klines(symbol, interval, start_str)
    
    # Extract only the closing prices and timestamps
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['close']].astype(float)
    return df

def classify_state(pct_change):
    """Discretizes continuous percentage changes into 5 distinct market states."""
    if pct_change > 0.005:    # > 0.5% gain
        return 'Strong_Up'
    elif pct_change > 0.001:  # 0.1% to 0.5% gain
        return 'Up'
    elif pct_change < -0.005: # > 0.5% drop
        return 'Strong_Down'
    elif pct_change < -0.001: # 0.1% to 0.5% drop
        return 'Down'
    else:                     # -0.1% to 0.1% change
        return 'Flat'

def build_markov_chain(df):
    """Calculates percentage changes, assigns states, and builds the Transition Matrix."""
    # Calculate period-to-period returns
    df['pct_change'] = df['close'].pct_change()
    df.dropna(inplace=True)
    
    # Map returns to our discrete states
    df['state'] = df['pct_change'].apply(classify_state)
    
    # Create a column for the previous period's state
    df['prior_state'] = df['state'].shift(1)
    df.dropna(inplace=True)
    
    # Count transitions from prior_state -> state
    transitions = df.groupby(['prior_state', 'state']).size().unstack(fill_value=0)
    
    # Convert absolute counts to row-wise probabilities (sum of each row = 1.0)
    transition_matrix = transitions.div(transitions.sum(axis=1), axis=0)
    return df, transition_matrix

def predict_next_state(current_state, transition_matrix):
    """Predicts the most likely next state based on the current state."""
    if current_state not in transition_matrix.index:
        return "Unknown", 0, {}
    
    # Isolate the row for the current state
    probabilities = transition_matrix.loc[current_state]
    
    # Find the state with the highest probability
    predicted_state = probabilities.idxmax()
    highest_prob = probabilities.max()
    
    return predicted_state, highest_prob, probabilities

def main():
    # 1. Fetch Market Data
    df = get_historical_data(SYMBOL, INTERVAL, LOOKBACK_DAYS)
    
    # 2. Build the Markov Model
    df, trans_matrix = build_markov_chain(df)
    
    print("\n--- Markov Chain Transition Probability Matrix ---")
    print(trans_matrix.round(3))
    
    # 3. Analyze Current Market Conditions
    current_state = df.iloc[-1]['state']
    last_close = df.iloc[-1]['close']
    
    # 4. Generate Prediction
    predicted_state, prob, all_probs = predict_next_state(current_state, trans_matrix)
    
    print("\n--- Prediction Output ---")
    print(f"Asset: {SYMBOL}")
    print(f"Latest Close Price: ${last_close:,.2f}")
    print(f"Current State (T=0): {current_state}")
    print(f"Predicted State (T+1): {predicted_state} (Confidence: {prob*100:.1f}%)")
    
    print("\nAll possible outcomes for the next candle:")
    for state, p in all_probs.items():
        print(f"  - {state}: {p*100:.1f}%")

if __name__ == "__main__":
    main()
