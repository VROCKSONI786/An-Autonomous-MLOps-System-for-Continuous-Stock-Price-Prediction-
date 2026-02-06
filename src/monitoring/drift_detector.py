import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
#from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
import logging
import os
import sys
import json
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """Detect data drift using EvidentlyAI"""
    
    def __init__(self, reference_period_days=90):
        self.reference_period_days = reference_period_days
        self.drift_reports = {}
        
    def load_data(self, filepath=None):
        """Load preprocessed data"""
        try:
            if filepath is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                filepath = os.path.join(project_root, "data", "processed", "preprocessed_data.csv")
            
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def split_reference_current(self, df, symbol):
        """Split data into reference and current windows"""
        logger.info(f"Splitting data for {symbol}")
        
        # Filter by symbol
        symbol_data = df[df['Symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_index()
        
        # Get date range
        end_date = symbol_data.index.max()
        reference_start = end_date - timedelta(days=self.reference_period_days * 2)
        reference_end = end_date - timedelta(days=self.reference_period_days)
        
        # Reference window (older data)
        reference_df = symbol_data[
            (symbol_data.index >= reference_start) & 
            (symbol_data.index < reference_end)
        ]
        
        # Current window (recent data)
        current_df = symbol_data[symbol_data.index >= reference_end]
        
        # Drop Symbol column for drift detection
        reference_df = reference_df.drop('Symbol', axis=1, errors='ignore')
        current_df = current_df.drop('Symbol', axis=1, errors='ignore')
        
        logger.info(f"Reference data: {len(reference_df)} rows")
        logger.info(f"Current data: {len(current_df)} rows")
        
        return reference_df, current_df
    
    def detect_drift(self, reference_df, current_df, symbol):
        """Detect drift between reference and current data"""
        logger.info(f"Detecting drift for {symbol}...")
        
        try:
            # Create drift report with updated API
            report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            # Run report
            report.run(
                reference_data=reference_df,
                current_data=current_df
            )
            
            # Get report as dict
            report_dict = report.as_dict()
            
            # Extract key metrics (API may vary by version)
            try:
                # Try newer API format
                drift_metrics = report_dict['metrics'][0]['result']
                drift_detected = drift_metrics.get('dataset_drift', False)
                drift_share = drift_metrics.get('share_of_drifted_columns', 0.0)
            except (KeyError, IndexError):
                # Fallback to older format
                drift_detected = False
                drift_share = 0.0
                logger.warning("Could not extract drift metrics, using defaults")
            
            logger.info(f"Drift detected: {drift_detected}")
            logger.info(f"Drift share: {drift_share:.4f}")
            
            # Save HTML report
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            reports_dir = os.path.join(project_root, "data", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(reports_dir, f"{symbol}_drift_report_{timestamp}.html")
            report.save_html(report_path)
            logger.info(f"Report saved to {report_path}")
            
            # Store results
            drift_result = {
                'symbol': symbol,
                'timestamp': timestamp,
                'drift_detected': drift_detected,
                'drift_score': drift_share,
                'report_path': report_path,
                'reference_size': len(reference_df),
                'current_size': len(current_df)
            }
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_column_drift(self, reference_df, current_df, symbol):
        """Analyze drift for individual columns"""
        logger.info(f"Analyzing column-level drift for {symbol}...")
        
        column_drifts = []
        
        for col in reference_df.columns:
            try:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(reference_df[col]):
                    continue
                
                report = Report(metrics=[
                    ColumnDriftMetric(column_name=col)
                ])
                
                report.run(
                    reference_data=reference_df,
                    current_data=current_df
                )
                
                report_dict = report.as_dict()
                
                # Extract drift info with error handling
                try:
                    drift_info = report_dict['metrics'][0]['result']
                    column_drifts.append({
                        'column': col,
                        'drift_detected': drift_info.get('drift_detected', False),
                        'drift_score': drift_info.get('drift_score', 0.0)
                    })
                except (KeyError, IndexError):
                    logger.warning(f"Could not extract drift info for {col}")
                    continue
                
            except Exception as e:
                logger.warning(f"Could not analyze drift for column {col}: {str(e)}")
                continue
        
        # Sort by drift score
        column_drifts = sorted(column_drifts, key=lambda x: x['drift_score'], reverse=True)
        
        logger.info(f"Analyzed {len(column_drifts)} columns")
        
        return column_drifts
    
    def check_all_symbols(self, symbols):
        """Check drift for all symbols"""
        logger.info("\n" + "="*60)
        logger.info("Starting Drift Detection for All Symbols")
        logger.info("="*60 + "\n")
        
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        all_results = []
        
        for symbol in symbols:
            try:
                logger.info(f"\n--- Checking {symbol} ---")
                
                # Split data
                reference_df, current_df = self.split_reference_current(df, symbol)
                
                if len(reference_df) == 0 or len(current_df) == 0:
                    logger.warning(f"Insufficient data for {symbol}, skipping...")
                    continue
                
                # Detect drift
                drift_result = self.detect_drift(reference_df, current_df, symbol)
                
                if drift_result:
                    # Analyze columns
                    column_drifts = self.analyze_column_drift(reference_df, current_df, symbol)
                    drift_result['column_drifts'] = column_drifts
                    
                    all_results.append(drift_result)
                    
                    # Log top drifting columns
                    if column_drifts:
                        logger.info("\nTop 5 drifting features:")
                        for col_drift in column_drifts[:5]:
                            logger.info(f"  {col_drift['column']}: {col_drift['drift_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save summary
        if all_results:
            self.save_drift_summary(all_results)
        
        logger.info("\n" + "="*60)
        logger.info("Drift Detection Complete")
        logger.info("="*60)
        
        return all_results
    
    def save_drift_summary(self, results):
        """Save drift detection summary"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        reports_dir = os.path.join(project_root, "data", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(reports_dir, f"drift_summary_{timestamp}.json")
        
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nDrift summary saved to {summary_path}")

if __name__ == "__main__":
    import yaml
    
    # Load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(project_root, "config", "config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        symbols = config['data_collection']['stock_symbols']
    except FileNotFoundError:
        logger.warning("Config file not found, using default symbols")
        symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Run drift detection
    detector = DriftDetector(reference_period_days=60)
    results = detector.check_all_symbols(symbols)
    
    # Print summary
    if results:
        print("\n" + "="*60)
        print("DRIFT DETECTION SUMMARY")
        print("="*60)
        for result in results:
            status = "DRIFT DETECTED" if result['drift_detected'] else "✓ No Drift"
            print(f"{result['symbol']}: {status} (Score: {result['drift_score']:.4f})")
    else:
        print("\nNo drift results generated. Check logs for errors.")