"""
Statistical Testing Framework for Adaptive Loss Systems
Comprehensive statistical validation and significance testing
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
from enum import Enum

class TestType(Enum):
    """Types of statistical tests"""
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non_parametric"
    PAIRED = "paired"
    INDEPENDENT = "independent"

@dataclass
class TestResult:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    assumptions_met: bool
    interpretation: str

class StatisticalTestSuite:
    """Comprehensive statistical testing suite"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}
    
    def test_normality(self, data: List[float], test_name: str = "shapiro") -> TestResult:
        """Test if data follows normal distribution"""
        data = np.array(data)
        
        if test_name == "shapiro":
            statistic, p_value = stats.shapiro(data)
        elif test_name == "kolmogorov":
            statistic, p_value = stats.kstest(data, 'norm')
        elif test_name == "anderson":
            result = stats.anderson(data, dist='norm')
            statistic = result.statistic
            p_value = result.significance_level[2]  # 5% significance level
        else:
            raise ValueError(f"Unknown normality test: {test_name}")
        
        return TestResult(
            test_name=f"Normality ({test_name})",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=0.0,  # Not applicable for normality tests
            confidence_interval=(0.0, 0.0),
            assumptions_met=True,
            interpretation="Data is normally distributed" if p_value >= self.alpha else "Data is not normally distributed"
        )
    
    def test_equal_variances(self, data_a: List[float], data_b: List[float]) -> TestResult:
        """Test if two datasets have equal variances"""
        data_a, data_b = np.array(data_a), np.array(data_b)
        
        # Levene's test
        statistic, p_value = stats.levene(data_a, data_b)
        
        return TestResult(
            test_name="Equal Variances (Levene)",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            assumptions_met=True,
            interpretation="Variances are equal" if p_value >= self.alpha else "Variances are not equal"
        )
    
    def compare_means(self, data_a: List[float], data_b: List[float], 
                     test_type: TestType = TestType.INDEPENDENT) -> TestResult:
        """Compare means between two datasets"""
        data_a, data_b = np.array(data_a), np.array(data_b)
        
        # Check normality
        norm_a = self.test_normality(data_a)
        norm_b = self.test_normality(data_b)
        
        # Check equal variances
        var_test = self.test_equal_variances(data_a, data_b)
        
        assumptions_met = not norm_a.significant and not norm_b.significant
        
        if assumptions_met and test_type == TestType.INDEPENDENT:
            # Parametric t-test
            equal_var = not var_test.significant
            statistic, p_value = stats.ttest_ind(data_a, data_b, equal_var=equal_var)
            test_name = f"Independent t-test (equal_var={equal_var})"
        elif test_type == TestType.PAIRED:
            # Paired t-test
            if len(data_a) != len(data_b):
                raise ValueError("Paired test requires equal sample sizes")
            statistic, p_value = stats.ttest_rel(data_a, data_b)
            test_name = "Paired t-test"
        else:
            # Non-parametric test
            statistic, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        
        # Effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(data_a, data_b)
        
        # Confidence interval for difference in means
        ci_lower, ci_upper = self._calculate_mean_difference_ci(data_a, data_b)
        
        return TestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            assumptions_met=assumptions_met,
            interpretation=self._interpret_comparison(data_a, data_b, p_value < self.alpha, effect_size)
        )
    
    def multiple_comparison_test(self, groups: Dict[str, List[float]]) -> Dict[str, TestResult]:
        """Perform multiple comparison tests"""
        group_names = list(groups.keys())
        group_data = list(groups.values())
        
        # Overall ANOVA or Kruskal-Wallis
        if len(group_names) < 2:
            return {}
        
        # Check normality for all groups
        all_normal = True
        for data in group_data:
            if self.test_normality(data).significant:
                all_normal = False
                break
        
        results = {}
        
        if all_normal:
            # One-way ANOVA
            statistic, p_value = stats.f_oneway(*group_data)
            results['overall'] = TestResult(
                test_name="One-way ANOVA",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.alpha,
                effect_size=0.0,  # Would need eta-squared calculation
                confidence_interval=(0.0, 0.0),
                assumptions_met=all_normal,
                interpretation="At least one group differs significantly" if p_value < self.alpha else "No significant differences between groups"
            )
        else:
            # Kruskal-Wallis test
            statistic, p_value = stats.kruskal(*group_data)
            results['overall'] = TestResult(
                test_name="Kruskal-Wallis",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.alpha,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                assumptions_met=True,
                interpretation="At least one group differs significantly" if p_value < self.alpha else "No significant differences between groups"
            )
        
        # Pairwise comparisons if overall test is significant
        if results['overall'].significant:
            for i, name_a in enumerate(group_names):
                for j, name_b in enumerate(group_names):
                    if i < j:
                        pair_result = self.compare_means(
                            groups[name_a], groups[name_b], 
                            TestType.NON_PARAMETRIC if not all_normal else TestType.INDEPENDENT
                        )
                        # Bonferroni correction
                        num_comparisons = len(group_names) * (len(group_names) - 1) // 2
                        corrected_alpha = self.alpha / num_comparisons
                        pair_result.significant = pair_result.p_value < corrected_alpha
                        
                        results[f"{name_a}_vs_{name_b}"] = pair_result
        
        return results
    
    def _calculate_cohens_d(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        var_a, var_b = np.var(data_a, ddof=1), np.var(data_b, ddof=1)
        n_a, n_b = len(data_a), len(data_b)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        return (mean_a - mean_b) / pooled_std
    
    def _calculate_mean_difference_ci(self, data_a: np.ndarray, data_b: np.ndarray, 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        var_a, var_b = np.var(data_a, ddof=1), np.var(data_b, ddof=1)
        n_a, n_b = len(data_a), len(data_b)
        
        # Standard error of difference
        se_diff = np.sqrt(var_a / n_a + var_b / n_b)
        
        # Degrees of freedom (Welch's formula)
        df = (var_a / n_a + var_b / n_b) ** 2 / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
        
        # Critical t-value
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        # Confidence interval
        diff = mean_a - mean_b
        margin_error = t_critical * se_diff
        
        return (diff - margin_error, diff + margin_error)
    
    def _interpret_comparison(self, data_a: np.ndarray, data_b: np.ndarray, 
                            significant: bool, effect_size: float) -> str:
        """Interpret comparison results"""
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        
        if not significant:
            return "No significant difference between groups"
        
        direction = "higher" if mean_a > mean_b else "lower"
        
        if abs(effect_size) < 0.2:
            magnitude = "small"
        elif abs(effect_size) < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return f"Group A is significantly {direction} than Group B ({magnitude} effect size)"

class ExperimentalValidation:
    """Experimental validation framework"""
    
    def __init__(self, alpha: float = 0.05):
        self.test_suite = StatisticalTestSuite(alpha)
        self.validation_results = {}
    
    def validate_experiment(self, baseline_results: List[float], 
                          adaptive_results: List[float],
                          experiment_name: str = "default") -> Dict[str, Any]:
        """Validate experimental results against baseline"""
        
        # Basic comparison
        comparison_result = self.test_suite.compare_means(
            baseline_results, adaptive_results, TestType.INDEPENDENT
        )
        
        # Descriptive statistics
        baseline_stats = self._calculate_descriptive_stats(baseline_results)
        adaptive_stats = self._calculate_descriptive_stats(adaptive_results)
        
        # Power analysis
        power_analysis = self._calculate_power_analysis(baseline_results, adaptive_results)
        
        # Practical significance
        practical_significance = self._assess_practical_significance(
            baseline_results, adaptive_results, comparison_result.effect_size
        )
        
        validation_result = {
            'experiment_name': experiment_name,
            'statistical_test': comparison_result,
            'baseline_stats': baseline_stats,
            'adaptive_stats': adaptive_stats,
            'power_analysis': power_analysis,
            'practical_significance': practical_significance,
            'recommendation': self._generate_recommendation(comparison_result, practical_significance)
        }
        
        self.validation_results[experiment_name] = validation_result
        return validation_result
    
    def _calculate_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics"""
        data = np.array(data)
        
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
    
    def _calculate_power_analysis(self, baseline: List[float], 
                                adaptive: List[float]) -> Dict[str, float]:
        """Calculate statistical power analysis"""
        effect_size = self.test_suite._calculate_cohens_d(np.array(baseline), np.array(adaptive))
        
        # This is a simplified power calculation
        # For more accurate results, you'd use specialized libraries like statsmodels
        alpha = self.test_suite.alpha
        n = min(len(baseline), len(adaptive))
        
        # Approximate power calculation for two-sample t-test
        delta = effect_size * np.sqrt(n / 2)
        t_critical = stats.t.ppf(1 - alpha / 2, 2 * n - 2)
        
        power = 1 - stats.t.cdf(t_critical - delta, 2 * n - 2) + stats.t.cdf(-t_critical - delta, 2 * n - 2)
        
        return {
            'effect_size': effect_size,
            'sample_size': n,
            'power': power,
            'adequate_power': power >= 0.8
        }
    
    def _assess_practical_significance(self, baseline: List[float], 
                                     adaptive: List[float], 
                                     effect_size: float) -> Dict[str, Any]:
        """Assess practical significance"""
        improvement = (np.mean(adaptive) - np.mean(baseline)) / np.mean(baseline)
        
        # Define practical significance thresholds
        if abs(improvement) < 0.05:
            practical_level = "negligible"
        elif abs(improvement) < 0.1:
            practical_level = "small"
        elif abs(improvement) < 0.2:
            practical_level = "moderate"
        else:
            practical_level = "large"
        
        return {
            'improvement_percentage': improvement * 100,
            'practical_level': practical_level,
            'effect_size': effect_size,
            'practically_significant': abs(improvement) >= 0.05
        }
    
    def _generate_recommendation(self, statistical_result: TestResult, 
                               practical_result: Dict[str, Any]) -> str:
        """Generate recommendation based on results"""
        if statistical_result.significant and practical_result['practically_significant']:
            return "Strong evidence for adaptive loss effectiveness - recommend adoption"
        elif statistical_result.significant and not practical_result['practically_significant']:
            return "Statistically significant but practically negligible - further investigation needed"
        elif not statistical_result.significant and practical_result['practically_significant']:
            return "Practically significant but not statistically significant - increase sample size"
        else:
            return "No evidence for adaptive loss effectiveness - not recommended"
    
    def generate_validation_report(self, experiment_name: str) -> str:
        """Generate comprehensive validation report"""
        if experiment_name not in self.validation_results:
            return "Experiment not found"
        
        result = self.validation_results[experiment_name]
        
        report = f"""
# Experimental Validation Report: {experiment_name}

## Statistical Test Results
- Test: {result['statistical_test'].test_name}
- Statistic: {result['statistical_test'].statistic:.4f}
- P-value: {result['statistical_test'].p_value:.4f}
- Significant: {result['statistical_test'].significant}
- Effect Size: {result['statistical_test'].effect_size:.4f}
- Interpretation: {result['statistical_test'].interpretation}

## Descriptive Statistics
### Baseline
- Mean: {result['baseline_stats']['mean']:.4f}
- Std: {result['baseline_stats']['std']:.4f}
- Median: {result['baseline_stats']['median']:.4f}

### Adaptive
- Mean: {result['adaptive_stats']['mean']:.4f}
- Std: {result['adaptive_stats']['std']:.4f}
- Median: {result['adaptive_stats']['median']:.4f}

## Power Analysis
- Effect Size: {result['power_analysis']['effect_size']:.4f}
- Power: {result['power_analysis']['power']:.4f}
- Adequate Power: {result['power_analysis']['adequate_power']}

## Practical Significance
- Improvement: {result['practical_significance']['improvement_percentage']:.2f}%
- Practical Level: {result['practical_significance']['practical_level']}
- Practically Significant: {result['practical_significance']['practically_significant']}

## Recommendation
{result['recommendation']}
"""
        
        return report
