"""
Data validation and cleaning pipeline for ARA AI
Implements missing data handling, outlier detection, and quality scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

from ara.core.exceptions import ValidationError
from ara.utils import get_logger

logger = get_logger(__name__)


class ImputationStrategy(Enum):
    """Strategies for handling missing data"""
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR_INTERPOLATE = "linear"
    SPLINE_INTERPOLATE = "spline"
    MEAN = "mean"
    MEDIAN = "median"
    DROP = "drop"


class OutlierMethod(Enum):
    """Methods for outlier detection"""
    IQR = "iqr"  # Interquartile Range
    Z_SCORE = "z_score"  # Standard deviations from mean
    MODIFIED_Z_SCORE = "modified_z_score"  # Using median absolute deviation
    ISOLATION_FOREST = "isolation_forest"  # ML-based


@dataclass
class DataQualityReport:
    """Report on data quality"""
    quality_score: float  # 0-1 scale
    total_rows: int
    missing_data: Dict[str, int]
    missing_percentage: float
    outliers_detected: Dict[str, int]
    outliers_percentage: float
    consistency_issues: List[str]
    recommendations: List[str]
    passed_validation: bool


@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    min_data_points: int = 30
    max_missing_percentage: float = 20.0
    outlier_method: OutlierMethod = OutlierMethod.IQR
    outlier_threshold: float = 3.0
    imputation_strategy: ImputationStrategy = ImputationStrategy.LINEAR_INTERPOLATE
    check_consistency: bool = True
    log_issues: bool = True


class DataValidator:
    """
    Validates and scores data quality
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
    
    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Validate dataframe and generate quality report
        
        Args:
            df: DataFrame to validate
            
        Returns:
            DataQualityReport with validation results
        """
        logger.info(f"Validating dataframe with {len(df)} rows")
        
        # Initialize report
        total_rows = len(df)
        issues = []
        recommendations = []
        
        # Check minimum data points
        if total_rows < self.config.min_data_points:
            issues.append(
                f"Insufficient data: {total_rows} rows "
                f"(minimum: {self.config.min_data_points})"
            )
            recommendations.append("Fetch more historical data")
        
        # Check missing data
        missing_data = self._check_missing_data(df)
        missing_pct = (
            sum(missing_data.values()) / (total_rows * len(df.columns)) * 100
        )
        
        if missing_pct > self.config.max_missing_percentage:
            issues.append(
                f"Too much missing data: {missing_pct:.1f}% "
                f"(max: {self.config.max_missing_percentage}%)"
            )
            recommendations.append("Use data imputation or fetch from alternative source")
        
        # Detect outliers
        outliers = self._detect_outliers(df)
        outliers_pct = (
            sum(outliers.values()) / (total_rows * len(df.columns)) * 100
        )
        
        # Check consistency
        consistency_issues = []
        if self.config.check_consistency:
            consistency_issues = self._check_consistency(df)
            if consistency_issues:
                issues.extend(consistency_issues)
                recommendations.append("Review data source for consistency issues")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            total_rows=total_rows,
            missing_pct=missing_pct,
            outliers_pct=outliers_pct,
            consistency_issues=len(consistency_issues)
        )
        
        # Determine if validation passed
        passed = (
            total_rows >= self.config.min_data_points and
            missing_pct <= self.config.max_missing_percentage and
            quality_score >= 0.5
        )
        
        # Log issues
        if self.config.log_issues and issues:
            for issue in issues:
                logger.warning(f"Data quality issue: {issue}")
        
        report = DataQualityReport(
            quality_score=quality_score,
            total_rows=total_rows,
            missing_data=missing_data,
            missing_percentage=missing_pct,
            outliers_detected=outliers,
            outliers_percentage=outliers_pct,
            consistency_issues=consistency_issues,
            recommendations=recommendations,
            passed_validation=passed
        )
        
        logger.info(
            f"Validation complete: score={quality_score:.2f}, passed={passed}",
            quality_score=quality_score,
            passed=passed
        )
        
        return report
    
    def _check_missing_data(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for missing data in each column
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict mapping column names to missing count
        """
        missing = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing[col] = missing_count
        
        return missing
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Detect outliers in numeric columns
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict mapping column names to outlier count
        """
        outliers = {}
        
        # Only check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['Volume']:  # Skip volume as it can vary widely
                continue
            
            if self.config.outlier_method == OutlierMethod.IQR:
                outlier_mask = self._detect_outliers_iqr(df[col])
            elif self.config.outlier_method == OutlierMethod.Z_SCORE:
                outlier_mask = self._detect_outliers_zscore(df[col])
            elif self.config.outlier_method == OutlierMethod.MODIFIED_Z_SCORE:
                outlier_mask = self._detect_outliers_modified_zscore(df[col])
            else:
                outlier_mask = self._detect_outliers_iqr(df[col])
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                outliers[col] = outlier_count
        
        return outliers
    
    def _detect_outliers_iqr(
        self,
        series: pd.Series,
        multiplier: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers using Interquartile Range method
        
        Args:
            series: Data series
            multiplier: IQR multiplier (default: 1.5)
            
        Returns:
            Boolean series indicating outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(
        self,
        series: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        Detect outliers using Z-score method
        
        Args:
            series: Data series
            threshold: Z-score threshold (default: from config)
            
        Returns:
            Boolean series indicating outliers
        """
        threshold = threshold or self.config.outlier_threshold
        
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.Series([False] * len(series), index=series.index)
        
        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold
    
    def _detect_outliers_modified_zscore(
        self,
        series: pd.Series,
        threshold: float = 3.5
    ) -> pd.Series:
        """
        Detect outliers using Modified Z-score (MAD-based)
        More robust to outliers than standard Z-score
        
        Args:
            series: Data series
            threshold: Modified Z-score threshold
            
        Returns:
            Boolean series indicating outliers
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            return pd.Series([False] * len(series), index=series.index)
        
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    def _check_consistency(self, df: pd.DataFrame) -> List[str]:
        """
        Check for data consistency issues
        
        Args:
            df: DataFrame to check
            
        Returns:
            List of consistency issues found
        """
        issues = []
        
        # Check if required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return issues  # Can't continue without these
        
        # Check High >= Low
        invalid_hl = (df['High'] < df['Low']).sum()
        if invalid_hl > 0:
            issues.append(f"High < Low in {invalid_hl} rows")
        
        # Check High >= Open, Close
        invalid_h_open = (df['High'] < df['Open']).sum()
        invalid_h_close = (df['High'] < df['Close']).sum()
        if invalid_h_open > 0 or invalid_h_close > 0:
            issues.append(
                f"High < Open/Close in {invalid_h_open + invalid_h_close} rows"
            )
        
        # Check Low <= Open, Close
        invalid_l_open = (df['Low'] > df['Open']).sum()
        invalid_l_close = (df['Low'] > df['Close']).sum()
        if invalid_l_open > 0 or invalid_l_close > 0:
            issues.append(
                f"Low > Open/Close in {invalid_l_open + invalid_l_close} rows"
            )
        
        # Check for negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"Negative values in {col}: {negative_count} rows")
        
        # Check for zero or negative volume
        zero_volume = (df['Volume'] <= 0).sum()
        if zero_volume > len(df) * 0.1:  # More than 10%
            issues.append(f"Zero/negative volume in {zero_volume} rows")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            issues.append(f"Duplicate timestamps: {dup_count} rows")
        
        # Check for large gaps in time series
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            median_diff = time_diffs.median()
            large_gaps = (time_diffs > median_diff * 3).sum()
            if large_gaps > 0:
                issues.append(f"Large time gaps detected: {large_gaps} instances")
        
        return issues
    
    def _calculate_quality_score(
        self,
        total_rows: int,
        missing_pct: float,
        outliers_pct: float,
        consistency_issues: int
    ) -> float:
        """
        Calculate overall data quality score (0-1)
        
        Args:
            total_rows: Total number of rows
            missing_pct: Percentage of missing data
            outliers_pct: Percentage of outliers
            consistency_issues: Number of consistency issues
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Penalize missing data (max -0.4)
        score -= min(0.4, missing_pct / 100 * 2)
        
        # Penalize outliers (max -0.2)
        score -= min(0.2, outliers_pct / 100)
        
        # Penalize consistency issues (max -0.3)
        score -= min(0.3, consistency_issues * 0.05)
        
        # Penalize insufficient data (max -0.1)
        if total_rows < self.config.min_data_points:
            score -= 0.1
        
        return max(0.0, min(1.0, score))


class DataCleaner:
    """
    Cleans and preprocesses data
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize cleaner
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.validator = DataValidator(config)
    
    def clean(
        self,
        df: pd.DataFrame,
        validate_first: bool = True
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Clean dataframe
        
        Args:
            df: DataFrame to clean
            validate_first: Run validation before cleaning
            
        Returns:
            Tuple of (cleaned DataFrame, quality report)
        """
        logger.info(f"Cleaning dataframe with {len(df)} rows")
        
        # Validate first if requested
        if validate_first:
            initial_report = self.validator.validate(df)
            logger.info(
                f"Initial quality score: {initial_report.quality_score:.2f}"
            )
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing data
        df_clean = self._handle_missing_data(df_clean)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Fix consistency issues
        df_clean = self._fix_consistency_issues(df_clean)
        
        # Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Sort by index (timestamp)
        df_clean = df_clean.sort_index()
        
        # Final validation
        final_report = self.validator.validate(df_clean)
        
        logger.info(
            f"Cleaning complete: {len(df)} -> {len(df_clean)} rows, "
            f"quality: {final_report.quality_score:.2f}",
            original_rows=len(df),
            cleaned_rows=len(df_clean),
            quality_score=final_report.quality_score
        )
        
        return df_clean, final_report
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data using configured strategy
        
        Args:
            df: DataFrame with missing data
            
        Returns:
            DataFrame with missing data handled
        """
        strategy = self.config.imputation_strategy
        
        if strategy == ImputationStrategy.FORWARD_FILL:
            df = df.fillna(method='ffill')
        
        elif strategy == ImputationStrategy.BACKWARD_FILL:
            df = df.fillna(method='bfill')
        
        elif strategy == ImputationStrategy.LINEAR_INTERPOLATE:
            df = df.interpolate(method='linear', limit_direction='both')
        
        elif strategy == ImputationStrategy.SPLINE_INTERPOLATE:
            # Spline requires at least 4 points
            if len(df) >= 4:
                df = df.interpolate(method='spline', order=3, limit_direction='both')
            else:
                df = df.interpolate(method='linear', limit_direction='both')
        
        elif strategy == ImputationStrategy.MEAN:
            df = df.fillna(df.mean())
        
        elif strategy == ImputationStrategy.MEDIAN:
            df = df.fillna(df.median())
        
        elif strategy == ImputationStrategy.DROP:
            df = df.dropna()
        
        # Final forward fill for any remaining NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers (clip to reasonable bounds)
        
        Args:
            df: DataFrame with potential outliers
            
        Returns:
            DataFrame with outliers handled
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['Volume']:  # Don't clip volume
                continue
            
            # Detect outliers
            if self.config.outlier_method == OutlierMethod.IQR:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            else:
                # Use mean +/- 3 std as fallback
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
            
            # Clip outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _fix_consistency_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix data consistency issues
        
        Args:
            df: DataFrame with potential consistency issues
            
        Returns:
            DataFrame with issues fixed
        """
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return df
        
        # Ensure High is the maximum
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        
        # Ensure Low is the minimum
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        # Ensure no negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df[col] = df[col].clip(lower=0)
        
        # Ensure no negative volume
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].clip(lower=0)
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate timestamps
        
        Args:
            df: DataFrame with potential duplicates
            
        Returns:
            DataFrame without duplicates
        """
        if df.index.duplicated().any():
            # Keep last occurrence
            df = df[~df.index.duplicated(keep='last')]
            logger.info("Removed duplicate timestamps")
        
        return df


class DataQualityScorer:
    """
    Scores data quality across multiple sources
    """
    
    @staticmethod
    def score_source(
        df: pd.DataFrame,
        source_name: str
    ) -> Dict[str, Any]:
        """
        Score data quality for a single source
        
        Args:
            df: DataFrame from source
            source_name: Name of the source
            
        Returns:
            Dict with quality metrics
        """
        validator = DataValidator()
        report = validator.validate(df)
        
        return {
            'source': source_name,
            'quality_score': report.quality_score,
            'total_rows': report.total_rows,
            'missing_percentage': report.missing_percentage,
            'outliers_percentage': report.outliers_percentage,
            'passed_validation': report.passed_validation,
            'issues': len(report.consistency_issues)
        }
    
    @staticmethod
    def compare_sources(
        sources: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Compare quality across multiple sources
        
        Args:
            sources: Dict mapping source names to DataFrames
            
        Returns:
            List of quality scores sorted by quality
        """
        scores = []
        
        for source_name, df in sources.items():
            score = DataQualityScorer.score_source(df, source_name)
            scores.append(score)
        
        # Sort by quality score (descending)
        scores.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return scores
    
    @staticmethod
    def select_best_source(
        sources: Dict[str, pd.DataFrame],
        min_quality: float = 0.7
    ) -> Tuple[str, pd.DataFrame]:
        """
        Select best quality source
        
        Args:
            sources: Dict mapping source names to DataFrames
            min_quality: Minimum acceptable quality score
            
        Returns:
            Tuple of (source name, DataFrame)
            
        Raises:
            ValidationError: If no source meets minimum quality
        """
        scores = DataQualityScorer.compare_sources(sources)
        
        if not scores:
            raise ValidationError("No data sources provided")
        
        best = scores[0]
        
        if best['quality_score'] < min_quality:
            raise ValidationError(
                f"Best source quality {best['quality_score']:.2f} "
                f"below minimum {min_quality}",
                {"scores": scores}
            )
        
        logger.info(
            f"Selected best source: {best['source']} "
            f"(quality: {best['quality_score']:.2f})"
        )
        
        return best['source'], sources[best['source']]
