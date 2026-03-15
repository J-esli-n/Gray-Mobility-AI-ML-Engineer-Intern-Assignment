def compute_risk(hr, spo2, bp_sys, anomaly):

    score = 0

    if spo2 < 92:
        score += 3

    if hr > 120:
        score += 2

    if bp_sys < 90:
        score += 2

    if anomaly == 1:
        score += 3

    if score <= 2:
        level = "LOW"
    elif score <= 5:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return score, level