"""Statistical analysis for benchmark results."""

import math
import statistics


class StatisticalAnalyzer:
    """Provides statistical analysis functions for benchmark data."""

    def mean(self, values: list[float]) -> float:
        """Calculate arithmetic mean."""
        if not values:
            return 0.0
        return statistics.mean(values)

    def median(self, values: list[float]) -> float:
        """Calculate median."""
        if not values:
            return 0.0
        return statistics.median(values)

    def std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        return statistics.stdev(values)

    def variance(self, values: list[float]) -> float:
        """Calculate variance."""
        if len(values) < 2:
            return 0.0
        return statistics.variance(values)

    def percentiles(
        self, values: list[float], percentiles: list[int]
    ) -> dict[int, float]:
        """Calculate multiple percentiles."""
        if not values:
            return dict.fromkeys(percentiles, 0.0)

        sorted_values = sorted(values)
        result = {}

        for p in percentiles:
            if p < 0 or p > 100:
                raise ValueError(f"Percentile must be between 0 and 100, got {p}")

            # Use linear interpolation for percentiles
            k = (len(sorted_values) - 1) * (p / 100.0)
            f = int(k)
            c = f + 1 if f < len(sorted_values) - 1 else f

            if f == c:
                result[p] = sorted_values[f]
            else:
                d0 = sorted_values[f] * (c - k)
                d1 = sorted_values[c] * (k - f)
                result[p] = d0 + d1

        return result

    def iqr(self, values: list[float]) -> float:
        """Calculate interquartile range."""
        if not values:
            return 0.0

        percentiles_data = self.percentiles(values, [25, 75])
        return percentiles_data[75] - percentiles_data[25]

    def mad(self, values: list[float]) -> float:
        """Calculate median absolute deviation."""
        if not values:
            return 0.0

        median_val = self.median(values)
        deviations = [abs(v - median_val) for v in values]
        return self.median(deviations)

    def detect_outliers(self, values: list[float], method: str = "iqr") -> list[int]:
        """Detect outliers using specified method."""
        if not values:
            return []

        outlier_indices = []

        if method == "iqr":
            # IQR method
            q1, q3 = self.percentiles(values, [25, 75]).values()
            iqr_val = q3 - q1
            lower_bound = q1 - 1.5 * iqr_val
            upper_bound = q3 + 1.5 * iqr_val

            for i, v in enumerate(values):
                if v < lower_bound or v > upper_bound:
                    outlier_indices.append(i)

        elif method == "zscore":
            # Z-score method
            mean_val = self.mean(values)
            std_val = self.std_dev(values)

            if std_val > 0:
                for i, v in enumerate(values):
                    z_score = abs((v - mean_val) / std_val)
                    if z_score > 3:  # Common threshold
                        outlier_indices.append(i)

        elif method == "mad":
            # Median Absolute Deviation method
            median_val = self.median(values)
            mad_val = self.mad(values)

            if mad_val > 0:
                for i, v in enumerate(values):
                    # Modified z-score
                    modified_z = 0.6745 * (v - median_val) / mad_val
                    if abs(modified_z) > 3.5:
                        outlier_indices.append(i)

        return outlier_indices

    def coefficient_of_variation(self, values: list[float]) -> float:
        """Calculate coefficient of variation (CV)."""
        mean_val = self.mean(values)
        if mean_val == 0:
            return 0.0

        std_val = self.std_dev(values)
        return std_val / abs(mean_val)

    def confidence_interval(
        self, values: list[float], confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval using t-distribution."""
        if len(values) < 2:
            return (0.0, 0.0)

        import math

        mean_val = self.mean(values)
        std_err = self.std_dev(values) / math.sqrt(len(values))

        # For simplicity, using z-score approximation
        # For small samples, should use t-distribution
        z_scores = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}

        z = z_scores.get(confidence, 1.960)
        margin = z * std_err

        return (mean_val - margin, mean_val + margin)

    def moving_average(self, values: list[float], window: int) -> list[float]:
        """Calculate moving average."""
        if not values or window <= 0:
            return []

        if window > len(values):
            window = len(values)

        result = []
        for i in range(len(values) - window + 1):
            window_values = values[i : i + window]
            result.append(self.mean(window_values))

        return result

    def exponential_moving_average(
        self, values: list[float], alpha: float = 0.3
    ) -> list[float]:
        """Calculate exponential moving average."""
        if not values:
            return []

        ema = [values[0]]

        for i in range(1, len(values)):
            ema_val = alpha * values[i] + (1 - alpha) * ema[-1]
            ema.append(ema_val)

        return ema

    def trend_direction(self, values: list[float], window: int = 5) -> str:
        """Detect trend direction using moving averages."""
        if len(values) < window * 2:
            return "insufficient_data"

        ma = self.moving_average(values, window)
        if not ma:
            return "no_trend"

        # Compare recent average with older average
        recent_avg = self.mean(ma[-window:])
        older_avg = self.mean(ma[:window])

        threshold = 0.05  # 5% change threshold
        change = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0

        if change > threshold:
            return "increasing"
        elif change < -threshold:
            return "decreasing"
        else:
            return "stable"

    def compare_distributions(
        self, dist1: list[float], dist2: list[float]
    ) -> dict[str, float | str]:
        """Compare two distributions."""
        if not dist1 or not dist2:
            return {"error": "Empty distribution(s)"}

        result = {
            "mean_diff": self.mean(dist2) - self.mean(dist1),
            "median_diff": self.median(dist2) - self.median(dist1),
            "std_dev_ratio": (
                self.std_dev(dist2) / self.std_dev(dist1)
                if self.std_dev(dist1) > 0
                else float("inf")
            ),
        }

        # Percentage changes
        if self.mean(dist1) != 0:
            result["mean_change_pct"] = (result["mean_diff"] / self.mean(dist1)) * 100

        if self.median(dist1) != 0:
            result["median_change_pct"] = (
                result["median_diff"] / self.median(dist1)
            ) * 100

        # Statistical significance (simplified)
        # For proper testing, use scipy.stats
        pooled_std = math.sqrt((self.variance(dist1) + self.variance(dist2)) / 2)

        if pooled_std > 0:
            effect_size = result["mean_diff"] / pooled_std
            result["effect_size"] = effect_size

            # Rough interpretation
            if abs(effect_size) < 0.2:
                result["effect"] = "negligible"
            elif abs(effect_size) < 0.5:
                result["effect"] = "small"
            elif abs(effect_size) < 0.8:
                result["effect"] = "medium"
            else:
                result["effect"] = "large"

        return result
