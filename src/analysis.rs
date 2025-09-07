/// Captures the four outcomes of binary classification to enable precision, recall, and accuracy calculations.
///
/// A confusion matrix is the fundamental data structure for evaluating binary classification
/// performance by categorizing predictions into true positives, false positives, true negatives,
/// and false negatives.
///
/// # Examples
///
/// ```rust
/// use policyai::analysis::ConfusionMatrix;
///
/// let mut matrix = ConfusionMatrix::new();
/// matrix.add_prediction(true, true);   // true positive
/// matrix.add_prediction(false, false); // true negative
/// matrix.add_prediction(true, false);  // false negative
/// matrix.add_prediction(false, true);  // false positive
///
/// assert_eq!(matrix.precision(), 0.5); // 1 TP / (1 TP + 1 FP)
/// assert_eq!(matrix.recall(), 0.5);    // 1 TP / (1 TP + 1 FN)
/// ```
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ConfusionMatrix {
    /// Count of correctly predicted positive cases (actual=true, predicted=true).
    pub true_positive: usize,
    /// Count of incorrectly predicted positive cases (actual=false, predicted=true).
    pub false_positive: usize,
    /// Count of correctly predicted negative cases (actual=false, predicted=false).
    pub true_negative: usize,
    /// Count of incorrectly predicted negative cases (actual=true, predicted=false).
    pub false_negative: usize,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix with all counts initialized to zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single prediction outcome by incrementing the appropriate confusion matrix cell.
    ///
    /// This method categorizes each prediction into one of four outcomes based on the
    /// combination of actual and predicted boolean values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use policyai::analysis::ConfusionMatrix;
    ///
    /// let mut matrix = ConfusionMatrix::new();
    /// matrix.add_prediction(true, true);   // Increment true_positive
    /// matrix.add_prediction(false, false); // Increment true_negative
    ///
    /// assert_eq!(matrix.true_positive, 1);
    /// assert_eq!(matrix.true_negative, 1);
    /// ```
    pub fn add_prediction(&mut self, actual: bool, predicted: bool) {
        match (actual, predicted) {
            (true, true) => self.true_positive += 1,
            (true, false) => self.false_negative += 1,
            (false, true) => self.false_positive += 1,
            (false, false) => self.true_negative += 1,
        }
    }

    /// Calculate precision as the fraction of positive predictions that were correct.
    ///
    /// Precision = True Positives / (True Positives + False Positives)
    ///
    /// Returns 0.0 when no positive predictions have been made.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use policyai::analysis::ConfusionMatrix;
    ///
    /// let mut matrix = ConfusionMatrix::new();
    /// matrix.true_positive = 8;
    /// matrix.false_positive = 2;
    ///
    /// assert_eq!(matrix.precision(), 0.8); // 8 / (8 + 2)
    /// ```
    pub fn precision(&self) -> f64 {
        let tp = self.true_positive as f64;
        let fp = self.false_positive as f64;
        if tp + fp == 0.0 {
            0.0
        } else {
            tp / (tp + fp)
        }
    }

    /// Calculate recall as the fraction of actual positive cases that were correctly identified.
    ///
    /// Recall = True Positives / (True Positives + False Negatives)
    ///
    /// Returns 0.0 when no actual positive cases exist.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use policyai::analysis::ConfusionMatrix;
    ///
    /// let mut matrix = ConfusionMatrix::new();
    /// matrix.true_positive = 6;
    /// matrix.false_negative = 2;
    ///
    /// assert_eq!(matrix.recall(), 0.75); // 6 / (6 + 2)
    /// ```
    pub fn recall(&self) -> f64 {
        let tp = self.true_positive as f64;
        let fn_count = self.false_negative as f64;
        if tp + fn_count == 0.0 {
            0.0
        } else {
            tp / (tp + fn_count)
        }
    }

    /// Calculate F1 score as the harmonic mean of precision and recall.
    ///
    /// F1 = 2 * (Precision * Recall) / (Precision + Recall)
    ///
    /// The F1 score balances precision and recall into a single metric, with values
    /// ranging from 0.0 (worst) to 1.0 (perfect). Returns 0.0 when both precision
    /// and recall are zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use policyai::analysis::ConfusionMatrix;
    ///
    /// let mut matrix = ConfusionMatrix::new();
    /// matrix.true_positive = 6;
    /// matrix.false_positive = 2;
    /// matrix.false_negative = 1;
    ///
    /// // Precision: 6/(6+2) = 0.75, Recall: 6/(6+1) ≈ 0.857
    /// // F1: 2 * 0.75 * 0.857 / (0.75 + 0.857) ≈ 0.8
    /// assert!((matrix.f1_score() - 0.8).abs() < 0.01);
    /// ```
    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Calculate accuracy as the fraction of all predictions that were correct.
    ///
    /// Accuracy = (True Positives + True Negatives) / Total Predictions
    ///
    /// Returns 0.0 when no predictions have been recorded.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use policyai::analysis::ConfusionMatrix;
    ///
    /// let mut matrix = ConfusionMatrix::new();
    /// matrix.true_positive = 4;
    /// matrix.true_negative = 3;
    /// matrix.false_positive = 2;
    /// matrix.false_negative = 1;
    ///
    /// assert_eq!(matrix.accuracy(), 0.7); // (4 + 3) / (4 + 3 + 2 + 1)
    /// ```
    pub fn accuracy(&self) -> f64 {
        let total =
            (self.true_positive + self.false_positive + self.true_negative + self.false_negative)
                as f64;
        if total == 0.0 {
            0.0
        } else {
            (self.true_positive + self.true_negative) as f64 / total
        }
    }
}

/// Aggregates performance metrics across multiple reports to compare PolicyAI against baseline extraction.
///
/// This structure accumulates field matching accuracy, error rates, and execution timing
/// across a collection of extraction reports, enabling comprehensive performance analysis
/// between PolicyAI and baseline extraction systems.
///
/// # Examples
///
/// ```rust
/// use policyai::analysis::RegressionAnalysis;
/// use policyai::data::Metrics;
///
/// let mut analysis = RegressionAnalysis::new();
///
/// let metrics = Metrics {
///     policyai_fields_matched: 8,
///     baseline_fields_matched: 6,
///     policyai_apply_duration_ms: 150,
///     baseline_apply_duration_ms: 200,
///     // ... other fields
/// #   policyai_fields_with_wrong_value: 0,
/// #   baseline_fields_with_wrong_value: 1,
/// #   policyai_fields_missing: 0,
/// #   baseline_fields_missing: 2,
/// #   policyai_extra_fields: 1,
/// #   baseline_extra_fields: 0,
/// #   policyai_error: None,
/// #   baseline_error: None,
/// #   policyai_usage: None,
/// #   baseline_usage: None,
/// };
///
/// analysis.add_report(&metrics);
/// assert_eq!(analysis.policyai_avg_fields_matched(), 8.0);
/// assert_eq!(analysis.baseline_avg_fields_matched(), 6.0);
/// ```
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RegressionAnalysis {
    /// Total number of reports processed in this analysis.
    pub total_reports: usize,
    /// Cumulative count of fields successfully matched by PolicyAI across all reports.
    pub policyai_total_fields_matched: usize,
    /// Cumulative count of fields successfully matched by baseline across all reports.
    pub baseline_total_fields_matched: usize,
    /// Cumulative count of fields with incorrect values extracted by PolicyAI.
    pub policyai_total_wrong_values: usize,
    /// Cumulative count of fields with incorrect values extracted by baseline.
    pub baseline_total_wrong_values: usize,
    /// Cumulative count of expected fields that PolicyAI failed to extract.
    pub policyai_total_missing_fields: usize,
    /// Cumulative count of expected fields that baseline failed to extract.
    pub baseline_total_missing_fields: usize,
    /// Cumulative count of unexpected fields extracted by PolicyAI.
    pub policyai_total_extra_fields: usize,
    /// Cumulative count of unexpected fields extracted by baseline.
    pub baseline_total_extra_fields: usize,
    /// Total number of reports where PolicyAI extraction encountered errors.
    pub policyai_errors: usize,
    /// Total number of reports where baseline extraction encountered errors.
    pub baseline_errors: usize,
    /// Total time in milliseconds spent on PolicyAI extraction across all reports.
    pub policyai_total_duration_ms: u64,
    /// Total time in milliseconds spent on baseline extraction across all reports.
    pub baseline_total_duration_ms: u64,
}

impl RegressionAnalysis {
    /// Create a new regression analysis with all metrics initialized to zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Incorporate metrics from a single report into the cumulative analysis.
    ///
    /// This method updates all relevant counters and totals with the values from
    /// the provided metrics, enabling aggregate analysis across multiple reports.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use policyai::analysis::RegressionAnalysis;
    /// use policyai::data::Metrics;
    ///
    /// let mut analysis = RegressionAnalysis::new();
    /// let metrics = Metrics {
    ///     policyai_fields_matched: 5,
    ///     baseline_fields_matched: 3,
    ///     policyai_apply_duration_ms: 100,
    ///     baseline_apply_duration_ms: 150,
    ///     // ... other fields
    /// #   policyai_fields_with_wrong_value: 0,
    /// #   baseline_fields_with_wrong_value: 1,
    /// #   policyai_fields_missing: 0,
    /// #   baseline_fields_missing: 1,
    /// #   policyai_extra_fields: 0,
    /// #   baseline_extra_fields: 0,
    /// #   policyai_error: None,
    /// #   baseline_error: None,
    /// #   policyai_usage: None,
    /// #   baseline_usage: None,
    /// };
    ///
    /// analysis.add_report(&metrics);
    /// assert_eq!(analysis.total_reports, 1);
    /// assert_eq!(analysis.policyai_total_fields_matched, 5);
    /// ```
    pub fn add_report(&mut self, metrics: &crate::data::Metrics) {
        self.total_reports += 1;
        self.policyai_total_fields_matched += metrics.policyai_fields_matched;
        self.baseline_total_fields_matched += metrics.baseline_fields_matched;
        self.policyai_total_wrong_values += metrics.policyai_fields_with_wrong_value;
        self.baseline_total_wrong_values += metrics.baseline_fields_with_wrong_value;
        self.policyai_total_missing_fields += metrics.policyai_fields_missing;
        self.baseline_total_missing_fields += metrics.baseline_fields_missing;
        self.policyai_total_extra_fields += metrics.policyai_extra_fields;
        self.baseline_total_extra_fields += metrics.baseline_extra_fields;

        if metrics.policyai_error.is_some() {
            self.policyai_errors += 1;
        }
        if metrics.baseline_error.is_some() {
            self.baseline_errors += 1;
        }

        self.policyai_total_duration_ms += metrics.policyai_apply_duration_ms as u64;
        self.baseline_total_duration_ms += metrics.baseline_apply_duration_ms as u64;
    }

    /// Calculate the average PolicyAI extraction duration per report in milliseconds.
    ///
    /// Returns 0.0 when no reports have been processed.
    pub fn policyai_avg_duration_ms(&self) -> f64 {
        if self.total_reports == 0 {
            0.0
        } else {
            self.policyai_total_duration_ms as f64 / self.total_reports as f64
        }
    }

    /// Calculate the average baseline extraction duration per report in milliseconds.
    ///
    /// Returns 0.0 when no reports have been processed.
    pub fn baseline_avg_duration_ms(&self) -> f64 {
        if self.total_reports == 0 {
            0.0
        } else {
            self.baseline_total_duration_ms as f64 / self.total_reports as f64
        }
    }

    /// Calculate the PolicyAI error rate as a fraction of total reports.
    ///
    /// Returns a value between 0.0 and 1.0, where 0.0 indicates no errors
    /// and 1.0 indicates errors in all reports. Returns 0.0 when no reports
    /// have been processed.
    pub fn policyai_error_rate(&self) -> f64 {
        if self.total_reports == 0 {
            0.0
        } else {
            self.policyai_errors as f64 / self.total_reports as f64
        }
    }

    /// Calculate the baseline error rate as a fraction of total reports.
    ///
    /// Returns a value between 0.0 and 1.0, where 0.0 indicates no errors
    /// and 1.0 indicates errors in all reports. Returns 0.0 when no reports
    /// have been processed.
    pub fn baseline_error_rate(&self) -> f64 {
        if self.total_reports == 0 {
            0.0
        } else {
            self.baseline_errors as f64 / self.total_reports as f64
        }
    }

    /// Calculate the average number of fields matched by PolicyAI per report.
    ///
    /// Returns 0.0 when no reports have been processed.
    pub fn policyai_avg_fields_matched(&self) -> f64 {
        if self.total_reports == 0 {
            0.0
        } else {
            self.policyai_total_fields_matched as f64 / self.total_reports as f64
        }
    }

    /// Calculate the average number of fields matched by baseline per report.
    ///
    /// Returns 0.0 when no reports have been processed.
    pub fn baseline_avg_fields_matched(&self) -> f64 {
        if self.total_reports == 0 {
            0.0
        } else {
            self.baseline_total_fields_matched as f64 / self.total_reports as f64
        }
    }
}

/// Applies confusion matrix analysis to field matching accuracy between PolicyAI and baseline.
///
/// This structure treats baseline performance as the "actual" values and PolicyAI performance
/// as the "predicted" values to evaluate whether PolicyAI correctly matches the expected
/// field count when baseline also matches correctly, and vice versa.
///
/// # Examples
///
/// ```rust
/// use policyai::analysis::FieldMatchAccuracyMatrix;
/// use policyai::data::Metrics;
///
/// let mut matrix = FieldMatchAccuracyMatrix::new();
/// let metrics = Metrics {
///     policyai_fields_matched: 5,
///     baseline_fields_matched: 5,
///     // ... other fields
/// #   policyai_fields_with_wrong_value: 0,
/// #   baseline_fields_with_wrong_value: 0,
/// #   policyai_fields_missing: 0,
/// #   baseline_fields_missing: 0,
/// #   policyai_extra_fields: 0,
/// #   baseline_extra_fields: 0,
/// #   policyai_error: None,
/// #   baseline_error: None,
/// #   policyai_apply_duration_ms: 100,
/// #   baseline_apply_duration_ms: 150,
/// #   policyai_usage: None,
/// #   baseline_usage: None,
/// };
///
/// matrix.add_report(&metrics, 5); // Both match expected count of 5
/// assert_eq!(matrix.confusion_matrix.true_positive, 1);
/// ```
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct FieldMatchAccuracyMatrix {
    /// Confusion matrix tracking field matching accuracy comparisons.
    pub confusion_matrix: ConfusionMatrix,
}

impl FieldMatchAccuracyMatrix {
    /// Create a new field match accuracy matrix with an empty confusion matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record field matching accuracy for both PolicyAI and baseline against expected field count.
    ///
    /// This method treats baseline correctness as the "actual" value and PolicyAI correctness
    /// as the "predicted" value, enabling analysis of whether PolicyAI performs similarly
    /// to baseline in terms of matching the expected number of fields.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use policyai::analysis::FieldMatchAccuracyMatrix;
    /// use policyai::data::Metrics;
    ///
    /// let mut matrix = FieldMatchAccuracyMatrix::new();
    /// let metrics = Metrics {
    ///     policyai_fields_matched: 4,  // Incorrect (expected 5)
    ///     baseline_fields_matched: 5,  // Correct
    ///     // ... other fields
    /// #   policyai_fields_with_wrong_value: 0,
    /// #   baseline_fields_with_wrong_value: 0,
    /// #   policyai_fields_missing: 0,
    /// #   baseline_fields_missing: 0,
    /// #   policyai_extra_fields: 0,
    /// #   baseline_extra_fields: 0,
    /// #   policyai_error: None,
    /// #   baseline_error: None,
    /// #   policyai_apply_duration_ms: 100,
    /// #   baseline_apply_duration_ms: 150,
    /// #   policyai_usage: None,
    /// #   baseline_usage: None,
    /// };
    ///
    /// matrix.add_report(&metrics, 5); // This creates a false negative
    /// assert_eq!(matrix.confusion_matrix.false_negative, 1);
    /// ```
    pub fn add_report(&mut self, metrics: &crate::data::Metrics, expected_field_count: usize) {
        // Actual = baseline correctly matches expected field count
        let baseline_correct = metrics.baseline_fields_matched == expected_field_count;
        // Predicted = PolicyAI correctly matches expected field count
        let policyai_correct = metrics.policyai_fields_matched == expected_field_count;

        self.confusion_matrix
            .add_prediction(baseline_correct, policyai_correct);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Metrics;

    #[test]
    fn confusion_matrix_new() {
        let matrix = ConfusionMatrix::new();
        assert_eq!(matrix.true_positive, 0);
        assert_eq!(matrix.false_positive, 0);
        assert_eq!(matrix.true_negative, 0);
        assert_eq!(matrix.false_negative, 0);
    }

    #[test]
    fn confusion_matrix_add_predictions() {
        let mut matrix = ConfusionMatrix::new();

        matrix.add_prediction(true, true);
        assert_eq!(matrix.true_positive, 1);

        matrix.add_prediction(true, false);
        assert_eq!(matrix.false_negative, 1);

        matrix.add_prediction(false, true);
        assert_eq!(matrix.false_positive, 1);

        matrix.add_prediction(false, false);
        assert_eq!(matrix.true_negative, 1);

        assert_eq!(matrix.true_positive, 1);
        assert_eq!(matrix.false_negative, 1);
        assert_eq!(matrix.false_positive, 1);
        assert_eq!(matrix.true_negative, 1);
    }

    #[test]
    fn confusion_matrix_precision() {
        let mut matrix = ConfusionMatrix::new();
        matrix.true_positive = 3;
        matrix.false_positive = 2;

        assert_eq!(matrix.precision(), 0.6);

        let empty_matrix = ConfusionMatrix::new();
        assert_eq!(empty_matrix.precision(), 0.0);
    }

    #[test]
    fn confusion_matrix_recall() {
        let mut matrix = ConfusionMatrix::new();
        matrix.true_positive = 3;
        matrix.false_negative = 1;

        assert_eq!(matrix.recall(), 0.75);

        let empty_matrix = ConfusionMatrix::new();
        assert_eq!(empty_matrix.recall(), 0.0);
    }

    #[test]
    fn confusion_matrix_f1_score() {
        let mut matrix = ConfusionMatrix::new();
        matrix.true_positive = 3;
        matrix.false_positive = 2;
        matrix.false_negative = 1;

        let precision = 3.0 / 5.0;
        let recall = 3.0 / 4.0;
        let expected_f1 = 2.0 * precision * recall / (precision + recall);

        assert!((matrix.f1_score() - expected_f1).abs() < 1e-10);

        let empty_matrix = ConfusionMatrix::new();
        assert_eq!(empty_matrix.f1_score(), 0.0);
    }

    #[test]
    fn confusion_matrix_accuracy() {
        let mut matrix = ConfusionMatrix::new();
        matrix.true_positive = 3;
        matrix.false_positive = 2;
        matrix.true_negative = 4;
        matrix.false_negative = 1;

        assert_eq!(matrix.accuracy(), 0.7);

        let empty_matrix = ConfusionMatrix::new();
        assert_eq!(empty_matrix.accuracy(), 0.0);
    }

    #[test]
    fn regression_analysis_new() {
        let analysis = RegressionAnalysis::new();
        assert_eq!(analysis.total_reports, 0);
        assert_eq!(analysis.policyai_total_fields_matched, 0);
        assert_eq!(analysis.baseline_total_fields_matched, 0);
        assert_eq!(analysis.policyai_errors, 0);
        assert_eq!(analysis.baseline_errors, 0);
    }

    #[test]
    fn regression_analysis_add_report() {
        let mut analysis = RegressionAnalysis::new();

        let metrics = Metrics {
            policyai_fields_matched: 5,
            baseline_fields_matched: 3,
            policyai_fields_with_wrong_value: 1,
            baseline_fields_with_wrong_value: 2,
            policyai_fields_missing: 0,
            baseline_fields_missing: 1,
            policyai_extra_fields: 2,
            baseline_extra_fields: 0,
            policyai_error: None,
            baseline_error: Some("error".to_string()),
            policyai_apply_duration_ms: 100,
            baseline_apply_duration_ms: 150,
            policyai_usage: None,
            baseline_usage: None,
        };

        analysis.add_report(&metrics);

        assert_eq!(analysis.total_reports, 1);
        assert_eq!(analysis.policyai_total_fields_matched, 5);
        assert_eq!(analysis.baseline_total_fields_matched, 3);
        assert_eq!(analysis.policyai_total_wrong_values, 1);
        assert_eq!(analysis.baseline_total_wrong_values, 2);
        assert_eq!(analysis.policyai_total_missing_fields, 0);
        assert_eq!(analysis.baseline_total_missing_fields, 1);
        assert_eq!(analysis.policyai_total_extra_fields, 2);
        assert_eq!(analysis.baseline_total_extra_fields, 0);
        assert_eq!(analysis.policyai_errors, 0);
        assert_eq!(analysis.baseline_errors, 1);
        assert_eq!(analysis.policyai_total_duration_ms, 100);
        assert_eq!(analysis.baseline_total_duration_ms, 150);
    }

    #[test]
    fn regression_analysis_averages() {
        let mut analysis = RegressionAnalysis::new();

        // Add two reports
        let metrics1 = Metrics {
            policyai_fields_matched: 4,
            baseline_fields_matched: 2,
            policyai_fields_with_wrong_value: 0,
            baseline_fields_with_wrong_value: 1,
            policyai_fields_missing: 1,
            baseline_fields_missing: 2,
            policyai_extra_fields: 0,
            baseline_extra_fields: 0,
            policyai_error: Some("error".to_string()),
            baseline_error: None,
            policyai_apply_duration_ms: 200,
            baseline_apply_duration_ms: 300,
            policyai_usage: None,
            baseline_usage: None,
        };

        let metrics2 = Metrics {
            policyai_fields_matched: 6,
            baseline_fields_matched: 4,
            policyai_fields_with_wrong_value: 2,
            baseline_fields_with_wrong_value: 1,
            policyai_fields_missing: 0,
            baseline_fields_missing: 1,
            policyai_extra_fields: 1,
            baseline_extra_fields: 2,
            policyai_error: None,
            baseline_error: Some("error".to_string()),
            policyai_apply_duration_ms: 100,
            baseline_apply_duration_ms: 200,
            policyai_usage: None,
            baseline_usage: None,
        };

        analysis.add_report(&metrics1);
        analysis.add_report(&metrics2);

        assert_eq!(analysis.total_reports, 2);
        assert_eq!(analysis.policyai_avg_fields_matched(), 5.0); // (4 + 6) / 2
        assert_eq!(analysis.baseline_avg_fields_matched(), 3.0); // (2 + 4) / 2
        assert_eq!(analysis.policyai_avg_duration_ms(), 150.0); // (200 + 100) / 2
        assert_eq!(analysis.baseline_avg_duration_ms(), 250.0); // (300 + 200) / 2
        assert_eq!(analysis.policyai_error_rate(), 0.5); // 1 error out of 2 reports
        assert_eq!(analysis.baseline_error_rate(), 0.5); // 1 error out of 2 reports
    }

    #[test]
    fn confusion_matrix_serialization() {
        let mut matrix = ConfusionMatrix::new();
        matrix.true_positive = 1;
        matrix.false_positive = 2;
        matrix.true_negative = 3;
        matrix.false_negative = 4;

        let serialized = serde_json::to_string(&matrix).unwrap();
        let deserialized: ConfusionMatrix = serde_json::from_str(&serialized).unwrap();

        assert_eq!(matrix.true_positive, deserialized.true_positive);
        assert_eq!(matrix.false_positive, deserialized.false_positive);
        assert_eq!(matrix.true_negative, deserialized.true_negative);
        assert_eq!(matrix.false_negative, deserialized.false_negative);
    }

    #[test]
    fn regression_analysis_serialization() {
        let mut analysis = RegressionAnalysis::new();
        analysis.total_reports = 10;
        analysis.policyai_total_fields_matched = 50;
        analysis.baseline_total_fields_matched = 40;
        analysis.policyai_errors = 2;
        analysis.baseline_errors = 3;

        let serialized = serde_json::to_string(&analysis).unwrap();
        let deserialized: RegressionAnalysis = serde_json::from_str(&serialized).unwrap();

        assert_eq!(analysis.total_reports, deserialized.total_reports);
        assert_eq!(
            analysis.policyai_total_fields_matched,
            deserialized.policyai_total_fields_matched
        );
        assert_eq!(
            analysis.baseline_total_fields_matched,
            deserialized.baseline_total_fields_matched
        );
        assert_eq!(analysis.policyai_errors, deserialized.policyai_errors);
        assert_eq!(analysis.baseline_errors, deserialized.baseline_errors);
    }
}
