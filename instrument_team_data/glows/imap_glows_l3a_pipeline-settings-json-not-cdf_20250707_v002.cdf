{
    "description": "Settings for ground-processing pipeline for IMAP/GLOWS instrument",
    "version": "0.1",
    "date_of_creation_yyyymmdd": "20230527",
    "filter_based_on_daily_statistical_error": {
        "n_sigma_threshold_lower": 3.0,
        "n_sigma_threshold_upper": 3.0
    },
    "filter_based_on_comparison_of_spin_periods": {
        "relative_difference_threshold": 7.0e-5
    },
    "filter_based_on_temperature_std_dev": {
        "std_dev_threshold__celsius_deg": 2.03
    },
    "filter_based_on_hv_voltage_std_dev": {
        "std_dev_threshold__volt": 50.0
    },
    "filter_based_on_spin_period_std_dev": {
        "std_dev_threshold__sec": 0.033333
    },
    "filter_based_on_pulse_length_std_dev": {
        "std_dev_threshold__usec": 1.0
    },
    "filter_based_on_maps": {
        "angular_radius_for_excl_regions__deg": 2.0
    },
    "active_bad_time_flags": {
        "is_pps_missing": true,
        "is_time_status_missing": true,
        "is_phase_missing": true,
        "is_spin_period_missing": true,
        "is_overexposed": true,
        "is_direct_event_non_monotonic": true,
        "is_night": false,
        "is_hv_test_in_progress": true,
        "is_test_pulse_in_progress": true,
        "is_memory_error_detected": true,
        "is_generated_on_ground": true,
        "is_beyond_daily_statistical_error": true,
        "is_temperature_std_dev_beyond_threshold": true,
        "is_hv_voltage_std_dev_beyond_threshold": true,
        "is_spin_period_std_dev_beyond_threshold": true,
        "is_pulse_length_std_dev_beyond_threshold": true,
        "is_spin_period_difference_beyond_threshold": false
    },
    "active_bad_angle_flags": {
        "is_close_to_uv_source": true,
        "is_inside_excluded_region": true,
        "is_excluded_by_instr_team": true,
        "is_suspected_transient": true
    },
    "number_of_good_histograms_at_night": 3,
    "l3a_nominal_number_of_bins": 90
}