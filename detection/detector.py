def simple_detector(motion_level: float):
    """
    Fake logic:
    - If motion_level > 0.5 → 'fight_candidate'
    - Else → 'calm_or_idle'
    """
    if motion_level > 0.5:
        return {"pose": "fight_candidate", "confidence": motion_level}
    else:
        return {"pose": "calm_or_idle", "confidence": motion_level}


if __name__ == "__main__":
    # quick test
    test_values = [0.1, 0.4, 0.6, 0.9]
    for v in test_values:
        print(v, "→", simple_detector(v))
