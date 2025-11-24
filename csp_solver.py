class ConstraintSolver:
    def __init__(self, user_constraints):
        """
        user_constraints: dict
        {
            'max_risk_volatility': 0.05,
            'max_price_per_share': 200,
            'excluded_sectors': ['Energy', 'Tobacco']
        }
        """
        self.constraints = user_constraints

    def filter_universe(self, stock_metadata):
        """
        Input: List of stock dictionaries with metadata.
        Output: List of ticker strings that satisfy ALL constraints.
        """
        valid_tickers = []
        
        print(f"\n[CSP] Filtering {len(stock_metadata)} stocks against constraints...")
        
        for stock in stock_metadata:
            # 1. Price Constraint
            if stock['price'] > self.constraints.get('max_price_per_share', float('inf')):
                print(f" -> Rejecting {stock['ticker']}: Price {stock['price']} too high.")
                continue
                
            # 2. Ethical/Sector Constraint
            if stock['sector'] in self.constraints.get('excluded_sectors', []):
                print(f" -> Rejecting {stock['ticker']}: Sector {stock['sector']} excluded.")
                continue
                
            # 3. Volatility Constraint
            if stock['volatility'] > self.constraints.get('max_risk_volatility', 1.0):
                print(f" -> Rejecting {stock['ticker']}: Volatility {stock['volatility']} too high.")
                continue
                
            valid_tickers.append(stock['ticker'])
            
        print(f"[CSP] Final Universe: {valid_tickers}")
        return valid_tickers